#!/usr/bin/env python

# # **Welcome to LASER FUSION\***
#
# \***FU**LL **SI**MULATION IN **O**NE **N**OTEBOOK
#
# This notebook is an exercise in putting _all_ the required functionality for a LASER style model into a single notebook. We will use the results of this exercise to determine what functionality is common and can be extracted to features in the LASER Python package (e.g., demographics functions, spatial connectivity functions, sorted and static queues) and what functionality is specific to our northern Nigeria measles modeling and would be layered over LASER package functionality.
#
# Use `jupyter nbconvert --to script measles.ipynb` at the command line to convert this notebook to a single `.py` script.

# # Model
#
# We will create a blank object, called `model`, to hold the various pieces of information we have about our model.

# In[ ]:


from datetime import datetime

print(f"Running FUSION at {datetime.now(tz=None)}")  # noqa: DTZ005


class Model:
    pass


model = Model()


# ## Source Demographics Data
#
# We have some census data for Nigeria (2015) which we will use to set the initial populations for our nodes. Since we are modeling northern Nigeria, as a first cut we will only choose administrative districts which start with "NORTH_". That gives us 419 nodes with a total population of ~96M.

# In[ ]:


# setup initial populations
from pathlib import Path

import numpy as np
from nigeria import lgas

print(Path.cwd())
admin2 = {k: v for k, v in lgas.items() if len(k.split(":")) == 5}
print(f"{len(admin2)=}")

nn_nodes = {k: v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
print(f"{len(nn_nodes)=}")

initial_populations = np.array([v[0][0] for v in nn_nodes.values()])
print(f"{len(initial_populations)=}")
print(f"First 32 populations:\n{initial_populations[0:32]}")
print(f"{initial_populations.sum()=:,}")


# ## Parameters
#
# We need some parameters now. We will use `PropertySet` rather than a raw dictionary for the "syntactic sugar" of referencing `params.ticks` rather than `params["ticks"]` each time.
#
# Also, we will set the parameters separately as `meta_params` and `measles_params` but combine them into one parameter set for future use. We _could_ create `model.params = PropertySet({"meta":meta_params, "measles":measles_params})` and then reference them "by path" in the subsequent code, e.g., `params.meta.ticks` and `params.measles.inf_mean`.

# In[ ]:


from idmlaser.utils import PropertySet

meta_params = PropertySet(
    {
        "ticks": 3650,
        "cbr": 35,  # match KR value
        "output": Path.cwd() / "outputs",
        "prng_seed": 20240801,
        "eula_age": 5,
    }
)

measles_params = PropertySet(
    {
        "exp_mean": np.float32(7.0),
        "exp_std": np.float32(1.0),
        "inf_mean": np.float32(18.0),  # match KR value
        "inf_std": np.float32(2.0),  # match KR value
        "r_naught": np.float32(14.0),
        "seasonality_factor": np.float32(0.1),  # match KR value
        "seasonality_phase": np.float32(182.5),  # match KR value
    }
)

network_params = PropertySet(
    {
        "a": np.float32(1.0),  # population 1 power factor
        "b": np.float32(1.0),  # population 2 power factor
        "c": np.float32(2.0),  # distance power factor
        "k": np.float32(500.0),  # gravity constant  # match KR value
        "max_frac": np.float32(0.05),  # maximum fraction of population that can move in a single tick
    }
)

ri_params = PropertySet(
    {
        "ri_coverage": np.float32(0.75),
        "mcv1_start": int(8.5 * 365 / 12),
        "mcv1_end": int(9.5 * 365 / 12),
        "mcv2_start": int(14.5 * 365 / 12),
        "mcv2_end": int(15.5 * 365 / 12),
        "probability_mcv1_take": np.float32(0.85),
        "probability_mcv2_take": np.float32(0.95),
    }
)

model.params = PropertySet(meta_params, measles_params, network_params, ri_params)  # type: ignore
model.params.beta = model.params.r_naught / model.params.inf_mean  # type: ignore


# ## PRNG Seeding
#
# **_Note 1:_** We will use the `default_rng` from `numpy.random` for the model PRNG and `np.random` for the Numba PRNG (Numba rewrites np.random to its parallelized PRNG).
#
# **_Note 2:_** PRNG seeding for reproducibility depends on using the same number of cores (or enforced with `nb.set_num_threads()` or setting the `NUMBA_NUM_THREADS` environment variable?).

# In[ ]:


import numba as nb


@nb.njit(parallel=True)
def set_numba_seeds(seed: np.uint32):
    for t in nb.prange(nb.get_num_threads()):
        np.random.seed(seed + t)

    return


print(f"Setting the model PRNG and Numba PRNG seeds for {nb.get_num_threads()} threads using seed {model.params.prng_seed}")
model.prng = np.random.default_rng(np.uint32(model.params.prng_seed))
set_numba_seeds(np.uint32(model.prng.integers(0, 2**32)))


# ## Capacity Calculation
#
# We have our initial populations, but we need to allocate enough space to handle growth during the simulation.

# In[ ]:


from idmlaser.numpynumba import Population

capacity = initial_populations.sum()
print(f"initial {capacity=:,}")
print(f"{model.params.cbr=}, {model.params.ticks=}")  # type: ignore
growth = (1.0 + model.params.cbr / 1000) ** (model.params.ticks // 365)  # type: ignore
print(f"{growth=}")
capacity *= growth
capacity *= 1.01  # 1% buffer
capacity = np.uint32(np.round(capacity))
print(f"required {capacity=:,}")
print(f"Allocating capacity for {capacity:,} individuals")
population = Population(capacity)
model.population = population  # type: ignore
ifirst, ilast = population.add(initial_populations.sum())
print(f"{ifirst=:,}, {ilast=:,}")


# ## Node IDs
#
# Add a property for node IDs. 419 nodes requires 9 bits so we will allocate a 16 bit value. Negative IDs don't make sense, so, `uint16`.

# In[ ]:


population.add_scalar_property("nodeid", np.uint16)
index = 0
for nodeid, count in enumerate(initial_populations):
    population.nodeid[index : index + count] = nodeid
    index += count

print(
    f"Sample node IDs: {','.join([str(model.population.nodeid[i]) for i in range(0, initial_populations.sum(), initial_populations.sum() // 32)])}"
)


# ## Node Populations
#
# We will need the most recent population numbers in order to determine the births, based on CBR, for the upcoming year. We will also, later, use the current population to determine the effective force of infection, i.e., total contagion / node population.
#
# Default data type is `uint32`.

# In[ ]:


node_count = len(nn_nodes)
nodes = Population(capacity=node_count)  # TODO: rename `Population` to something appropriate to agents _and_ nodes
model.nodes = nodes  # type: ignore
ifirst, ilast = nodes.add(node_count)
print(f"{ifirst=:,}, {ilast=:,}")
nodes.add_vector_property("population", model.params.ticks + 1)  # type: ignore
nodes.population[:, 0] = initial_populations


# ## Population per Tick
#
# We will propagate the current populations forward on each tick. Vital dynamics of births and non-disease deaths will update the current values. The argument signature for per tick step phases is (`model`, `tick`). This lets functions access model specific properties and use the current tick, if necessary, e.g. record information or decide to act.

# In[ ]:


def propagate_population(model, tick):
    model.nodes.population[:, tick + 1] = model.nodes.population[:, tick]

    return


# ## Vital Dynamics: Births
#
# Let's implement births over time. We will use the CBR in `model.params` and draw for the number of births this year based on the most recent population. Then, we will distribute those births as evenly as possible for integral values over the days of the year.
#
# Note that we add in the date of birth and date of non-disease death after we add those properties below.
#
# Note that we add in initializing the susceptibility after we add that property below.

# In[ ]:


from idmlaser.kmcurve import pdsod

model.nodes.add_vector_property("births", (model.params.ticks + 364) // 365)  # births per year


def do_births(model, tick):
    doy = tick % 365 + 1  # day of year 1...365
    year = tick // 365

    if doy == 1:
        model.nodes.births[:, year] = model.prng.poisson(model.nodes.population[:, tick] * model.params.cbr / 1000)

    annual_births = model.nodes.births[:, year]
    todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
    count_births = todays_births.sum()
    istart, iend = model.population.add(count_births)  # add count_births agents to the population, get the indices of the new agents

    # enable this after loading the aliased distribution and dod and dob properties (see cells below)
    model.population.dob[istart:iend] = 0
    model.population.dod[istart:iend] = tick + pdsod(model.population.dob[istart:iend], max_year=100, prng=model.prng)
    model.population.dob[istart:iend] = tick  # now update dob to reflect being born today

    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 1

    index = istart
    nodeids = model.population.nodeid  # grab this once for efficiency
    dods = model.population.dod  # grab this once for efficiency
    max_tick = model.params.ticks
    for nodeid, births in enumerate(todays_births):
        nodeids[index : index + births] = nodeid
        for agent in range(index, index + births):
            # If the agent will die before the end of the simulation, add it to the queue
            if dods[agent] < max_tick:
                model.nddq.push(agent)
        index += births
    model.nodes.population[:, tick + 1] += todays_births

    # Draw RI timer values for newborns
    # TODO: Change or parameterize as appropriate
    ri_timer_values = model.prng.uniform(model.params.mcv1_start, model.params.mcv1_end, count_births).astype(np.uint16)

    # ri_timer is initialized to 0, we will just set the values for selected (fortunate) agents, based on their node RI coverage
    assert count_births == (iend - istart)
    mask = model.prng.random(count_births) < (model.nodes.ri_coverages[model.population.nodeid[istart:iend]])
    model.population.ri_timer[istart:iend][mask] = ri_timer_values[mask]

    # RI functions (see below)
    set_accessibility(model, istart, iend)
    add_with_ips(model, istart, iend)
    add_maternal_immunity(model, istart, iend)

    return


#
# # Non-Disease Mortality
#
# ## Non-Disease Mortality Part I
#
# We start by loading a population pyramid in order to initialize the ages of the initial population realistically.
#
# The population pyramid is typically in 5 year age buckets. Once we draw for the age bucket, we draw uniformly for a date of birth within the range of the bucket.
#
# **Note:** the values in `model.population.dob` are _positive_ at this point. Later we will negate them to convert them to dates of birth prior to now (t = 0).

# In[ ]:


from tqdm import tqdm

import idmlaser.pyramid as pyramid

if "__file__" not in globals():
    # running in a notebook _in the notebook directory_
    pyramid_file = Path.cwd().parent / "tests" / "USA-pyramid-2023.csv"
else:
    # running in a script in ??? directory, queue off the script directory
    pyramid_file = Path(__file__).parent / ".." / "tests" / "USA-pyramid-2023.csv"

print(f"Loading pyramid from '{pyramid_file}'...")
age_distribution = pyramid.load_pyramid_csv(pyramid_file)
print("Creating aliased distribution...")
aliased_distribution = pyramid.AliasedDistribution(age_distribution[:, 4], prng=model.prng)
count_active = initial_populations.sum()
print(f"Sampling {count_active:,} ages... {model.population.count=:,}")
buckets = aliased_distribution.sample(model.population.count)
print(f"First 32 buckets:\n{buckets[:32]}")
minimum_age = age_distribution[:, 0] * 365  # closed, include this value
limit_age = (age_distribution[:, 1] + 1) * 365  # open, exclude this value
model.population.add_scalar_property("dob", np.int32)
mask = np.zeros(capacity, dtype=bool)

print("Converting age buckets to ages...")
for i in tqdm(range(len(age_distribution))):
    mask[:count_active] = buckets == i  # indices of agents in this age group bucket
    # draw uniformly between the start and end of the age group bucket
    model.population.dob[mask] = model.prng.integers(low=minimum_age[i], high=limit_age[i], size=mask.sum())

print(f"First 32 ages:\n{model.population.dob[0:32]=}")


# ## Non-Disease Mortality Part II
#
# We import `pdsod` ("predicted_days_of_death") which currently uses a USA 2003 survival table. `pdsod` will draw for a year of death (in or after the current age year) and then will draw a day in the selected year.
#
# **Note:** the incoming values in `model.population.dob` are positive (_ages_). After we use them to draw for date of death, we negate them to convert them to dates of birth (in days) prior to now (t=0).

# In[ ]:


from idmlaser.kmcurve import pdsod  # noqa: F811

population.add_scalar_property("dod", np.int32)
dobs = population.dob[0 : population.count]
dods = population.dod[0 : population.count]
tstart = datetime.now(tz=None)  # noqa: DTZ005
dods[:] = pdsod(dobs[0 : population.count], max_year=100, prng=model.prng)
tfinish = datetime.now(tz=None)  # noqa: DTZ005
print(f"Elapsed time for drawing dates of death: {tfinish - tstart}")

dods -= dobs.astype(dods.dtype)  # renormalize to be relative to _now_ (t = 0)
dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative
print(f"First 32 DoBs (should all be negative - these agents were born before today):\n{dobs[:32]}")
print(f"First 32 DoDs (should all be positive - these agents will all pass in the future):\n{dods[:32]}")


# ### Interlude - EULAgize by Age
#
# Now that we have dates of birth (ages) and dates of non-disease death, we can collapse EULAs, chosen by age, into a scalar value for each node.
#
# Note that we could use other criteria, e.g., susceptibility _and_ age (i.e., we don't want to EULAgize newborns with maternal immunity). We just want to delay building the non-disease death priority queue until we know which agents we are keeping.

# In[ ]:


@nb.njit((nb.int64, nb.int64[:], nb.uint32[:], nb.uint32[:, :], nb.uint16[:], nb.int32[:]), parallel=True)
def collect_eula_data(count, indices, sizes, deaths, nodeids, dods):
    for i in nb.prange(count):
        index = indices[i]
        nodeid = nodeids[index]
        sizes[nodeid] += 1
        deaths[nodeid, dods[index]] += 1

    return


eulagize = model.population.dob[0 : model.population.count] <= (-model.params.eula_age * 365)
max_dod = model.population.dod.max()
print(f"Adding node property ('eula_sizes') to keep track of EULA size each tick {model.params.ticks+1}...")
model.nodes.add_vector_property("eula_sizes", model.params.ticks + 1)  # each node gets a 1D array of EULA sizes
print(f"Adding node property ('eula_deaths') to keep track of EULA deaths each day/tick {max_dod+1}...")
model.nodes.add_vector_property("eula_deaths", max_dod + 1)  # each node gets a 1D array of deaths by day of death
print("Collecting EULA data...")
collect_eula_data(
    eulagize.sum(),
    np.nonzero(eulagize)[0],
    model.nodes.eula_sizes[:, 0],
    model.nodes.eula_deaths,
    model.population.nodeid,
    model.population.dod,
)

print(f"Found {eulagize.sum():,} agents that are at least {model.params.eula_age} years old to EULAgize.")
print(f"EULA sizes for first eight nodes at t=0:\n{model.nodes.eula_sizes[0:8,0]}")
print(f"{model.nodes.eula_deaths.sum()=:,}")
print("First four days of EULA deaths for the first eight nodes:")
print(f"{model.nodes.eula_deaths[0:8,0:4]}")


# ### Squash Remaining Agents
#
# Now we need to squash the agents we will retain into the first N contiguous spaces of the various property vectors.

# In[ ]:


retain = ~eulagize
count = retain.sum()
print(f"Retaining {count:,} agents.")
model.population.squash(retain, verbose=True)


# Now we have the remaining active agents in `model.population`.
#
# We need a step function to update total node population with non-disease deaths in EULAs.

# In[ ]:


def do_eula_deaths(model, tick):
    # Previously we copied model.nodes.population[:, tick] forward into model.nodes.population[:, tick+1]
    # Remove newly deceased EULAs from the total population
    model.nodes.population[:, tick + 1] -= model.nodes.eula_deaths[:, tick]
    year = tick // 365
    model.nodes.deaths[:, year] += model.nodes.eula_deaths[:, tick]
    # We will remove newly deceased non-EULA agents from the population later.

    return


# ## Non-Disease Mortality Part III
#
# If we had no births, we could use a regular FIFO queue with the dates of death sorted to process non-disease deaths in time order. However, we may have agents born during the simulation who are predicted to die from non-disease causes before the end of the simulation and need to be insert, in sorted order, into the queue. So, we will use a priority queue - a data structure which allows for efficient insertion and removal while maintaining sorted order - to keep track of agents by date of non-disease death.

# In[ ]:


from idmlaser.utils import PriorityQueuePy

tstart = datetime.now(tz=None)  # noqa: DTZ005
dods = population.dod
mortality = PriorityQueuePy(capacity, dods)
for i in tqdm(indices := np.nonzero(dods[0 : population.count] < model.params.ticks)[0]):
    mortality.push(i)
tfinish = datetime.now(tz=None)  # noqa: DTZ005

print(f"Elapsed time for pushing dates of death: {tfinish - tstart}")
print(f"Non-disease mortality: tracked {len(indices):,}, untracked {population.count - len(indices):,}")

model.nddq = mortality


# ## Non-Disease Mortality Part IV
#
# During the simulation, we will process all agents with a predicated date of death today marking them as deceased and removing them from the active population count.
#
# **_TODO:_** Set a state to indicate "deceased" _or_ ignore the priority queue altogether and compare `dod` against `tick` and ignore agents where `dod < tick`.

# In[ ]:


model.nodes.add_vector_property("deaths", (model.params.ticks + 364) // 365)  # deaths per year


def do_non_disease_deaths(model, tick):
    nodeids = model.population.nodeid
    node_population = model.nodes.population[:, tick + 1]
    year = tick // 365

    node_deaths = model.nodes.deaths[:, year]
    susceptibility = model.population.susceptibility

    pq = model.nddq
    while (len(pq) > 0) and (pq.peekv() <= tick):
        i = pq.popi()
        nodeid = nodeids[i]
        node_population[nodeid] -= 1
        node_deaths[nodeid] += 1
        susceptibility[i] = 0  # disqualifies from infection etc.

    return


# ## Incubation and Infection
#
# We will add incubation timer (`etimer` for "exposure timer") and an infection (or infectious) timer (`itimer`) properties to the model population. A `uint8` counting down from as much as 255 days will be more than enough.
#
# We wrap a Numba compiled function using all available cores in the infection update function, extracting the `count` and `itimer` values the JITted function needs.
#
# Similarly, we wrap a Numba compiled function using all available cores in the exposure update function, extracting the values the JITted function needs from the `model` object.

# In[ ]:


model.population.add_scalar_property("etimer", np.uint8)
model.population.add_scalar_property("itimer", np.uint8)


@nb.njit((nb.uint32, nb.uint8[:]), parallel=True)
def _infection_update(count, itimers):
    for i in nb.prange(count):
        if itimers[i] > 0:
            itimers[i] -= 1

    return


def do_infection_update(model, tick):
    _infection_update(nb.uint32(model.population.count), model.population.itimer)

    return


@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
def _exposure_update(count, etimers, itimers, inf_mean, inf_std):
    for i in nb.prange(count):
        if etimers[i] > 0:
            etimers[i] -= 1
            if etimers[i] == 0:
                itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))  # must be at least 1 day

    return


def do_exposure_update(model, tick):
    _exposure_update(
        nb.uint32(model.population.count), model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std
    )

    return


# ## Susceptibility
#
# We add a `susceptibility` property to the population in order to identify susceptible agents.
#
# Maternal immunity, once implemented, will mean setting `susceptibility = 0` at birth and then setting `susceptibility = 1` at some point in the future when maternal antibodies have been determined to have waned.
#
# RI, once implemented, will set `susceptibility = 0` if the infant is determined to get an efficacious innoculation. The infant might _not_ get an innoculation depending on RI coverage in their home node. Also, an RI innoculation might not be efficacious due to interference by maternal antibodies or just by vaccine failure.
#
# Transmission, once implemented, will set `susceptibility = 0` so the agent is not considered for future transmission. I.e., agents in the exposed, infectious, recovered, or deceased state should have `susceptibility` set to 0.
#
# **Note:** We initialize `susceptibility` to 0 since the majority of the initial population will have already been exposed to measles or have been vaccinated. First cut here is to set all agents under age 5 as susceptible.
#
# **_TODO:_** Update this function to consider maternal immunity. Children under 1(?) year of age should draw for a date of maternal immunity waning. If they are _under_ that age, their susceptibility should be set to `0` and they should be put in a priority queue based on waning date to have their susceptibility set to `1`.
#
# **_TODO:_** Update this function to probabilistically set susceptibility to `0` for children over 1 year of age based on local RI coverage and other factors (estimated prevalence and probability of _not_ having had measles?).

# In[ ]:


model.population.add_scalar_property("susceptibility", np.uint8)


# initialize susceptibility based on age
@nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True)
def initialize_susceptibility(count, dob, susceptibility):
    for i in nb.prange(count):
        if dob[i] >= -365 * 5:
            susceptibility[i] = 1

    return


initialize_susceptibility(model.population.count, model.population.dob, model.population.susceptibility)


# # Routine Immunization (RI)
#
# Do MCV1 for most kids at around 9 months. Do MCV2 for most of the kids who didn't take around 15 months. There are a bunch of ways of doing this. What we're doing here is using an IP -- to borrow EMOD terminology -- for Accessibility, with values 0 (Easy), 1 (Medium), 2 (Hard). See elsewhere for that. The MCV1 timer is a uniform draw from 8.5 months to 9.5 months. MCV2 timer is a uniform draw from 14.5 months to 15.5 months.*
#
# *See RI parameters near the top.

# ## Coverages
#
# All MCV is subject to coverage levels which vary by node.

# In[ ]:


# Add new property "ri_coverages", just randomly for demonstration purposes
# Replace with values from data
model.nodes.add_scalar_property("ri_coverages", dtype=np.float32)
model.nodes.ri_coverages = model.prng.random(node_count)

print(f"RI coverage in first 32 nodes {model.nodes.ri_coverages[0:32]=}")


# ## Accessibility "IP" Groups
#
# - 85% of in-coverage kids get MCV1
# - 14% get MCV2
# - 1% get nothing
#
# Vary as needed (see RI parameters near top).

# In[ ]:


model.population.add_scalar_property("accessibility", np.uint8)

GET_MCV1 = 0
GET_MCV2 = 1
GET_NONE = 2


def set_accessibility(model, istart, iend):
    # determine the accessibilty thresholds _by node_ - could be done once if coverage is constant over time
    mcv1_cutoff = model.nodes.ri_coverages * model.params.probability_mcv1_take  # probability of (MCV1 vaccination) _and_ (MCV1 take)
    mcv2_cutoff = (
        mcv1_cutoff + model.nodes.ri_coverages * (1.0 - model.params.probability_mcv1_take) * model.params.probability_mcv2_take
    )  # probabilty of (MCV1 vaccinaton) _and_ (not MCV1 take) and (MCV2 take)

    # Generate random numbers to decide accessibility based on coverages
    random_values = model.prng.random(size=(iend - istart))

    # Set values based on attenuated probabilities
    nodeids = model.population.nodeid[istart:iend]  # get slice of nodeids for the new agents
    get_mcv2 = (random_values > mcv1_cutoff[nodeids]) & (random_values <= mcv2_cutoff[nodeids])
    get_none = random_values > mcv2_cutoff[nodeids]
    accessibility = model.population.accessibility[istart:iend]  # get slice of accessibility for the new agents
    accessibility[get_mcv2] = GET_MCV2
    accessibility[get_none] = GET_NONE

    return


# ## Set RI Timers
#
# Set newborns' `ri_timer` values vased on their accessibility property.

# In[ ]:


model.population.add_scalar_property("ri_timer", np.uint16)  # Use uint16 for timer since 15 months = 450 days > 2^8


def add_with_ips(model, istart, iend):
    # Define the timer ranges as needed
    count = iend - istart
    ri_timer_values_9mo = model.prng.integers(model.params.mcv1_start, model.params.mcv1_end, count).astype(np.uint16)
    ri_timer_values_15mo = model.prng.integers(model.params.mcv2_start, model.params.mcv2_end, count).astype(np.uint16)

    # Get slice of the accessibility values for the current agents
    accessibility = model.population.accessibility[istart:iend]

    # Masks based on accessibility
    mask_mcv1 = accessibility == GET_MCV1
    mask_mcv2 = accessibility == GET_MCV2
    mask_none = accessibility == GET_NONE  # for validation
    if mask_mcv2.sum() == 0:
        raise ValueError("Didn't find anyone with accessibility set to 1 (medium).")
    if mask_none.sum() == 0:
        raise ValueError("Didn't find anyone with accessibility set to 2 (medium).")

    # mask_none is unnecessary since we don't apply any timer for it

    # Apply the first RI/MCV1 timer to accessibility GET_MCV1
    timers = model.population.ri_timer[istart:iend]  # get slice of the RI timers for the new agents
    timers[mask_mcv1] = ri_timer_values_9mo[mask_mcv1]

    # Apply the second RI/MCV2 timer to accessibility GET_MCV2
    timers[mask_mcv2] = ri_timer_values_15mo[mask_mcv2]

    # No need to apply anything to accessibility GET_NONE, as it should remain unmodified

    return


# ## Update RI Timers on Tick

# In[ ]:


# Define the function to decrement ri_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility):
    for i in nb.prange(count):
        timer = ri_timer[i]
        if timer > 0:
            timer -= 1
            ri_timer[i] = timer
            # TBD: It's perfectly possible that the individual got infected (or recovered) while this timer
            # was counting down and we might want to abort the timer.
            if timer == 0:
                susceptibility[i] = 0


def do_ri(model, tick):
    _update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility)
    return


# # Maternal Immunity
#
# All newborns come into the world with susceptibility=0. They get a 6 month timer. When that timer hits 0, they become susceptible.
#
# **_Note:_** This interacts well with RI - RI too early sets susceptibility to 0, but the waning maternal immunity timer goes off _later_ and (re) sets susceptibility to 1.

# ## Initialize Maternal Immunity for Newborns

# In[ ]:


model.population.add_scalar_property("susceptibility_timer", np.uint8)  # 6 months in days ~180 < 2^8


# Add for newborns
def add_maternal_immunity(model, istart, iend):
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0  # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(6 * 365 / 12)  # 6 months in days

    return


@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_sus_timer(count, susceptibility_timer, susceptibility):
    for i in nb.prange(count):
        if susceptibility_timer[i] > 0:
            susceptibility_timer[i] -= 1
            if susceptibility_timer[i] == 0:
                susceptibility[i] = 1

    return


def do_susceptibility_decay(model, tick):
    _update_susceptibility_based_on_sus_timer(
        model.population.count, model.population.susceptibility_timer, model.population.susceptibility
    )

    return


# # Initial Infections
#
# We choose up to 10 infections per node initially. Another option would be to choose the initial number of infections based on estimated prevalence.
#
# **Note:** this function has a known bug based on using Numba to optimize the function. [Numba only exposes an atomic decrement in CUDA implementations](https://numba.readthedocs.io/en/stable/cuda/intrinsics.html) so on the CPU it is possible that more than one thread decides there are still agents to initialize as infected and the total count of initial infections is more than specified.
#
# **_TODO:_** After checking to see if there are still infections to be initialized in this node, consider checking the susceptibility of the agent as well, i.e., only infect susceptible agents.
#
# **_TODO_:** Fix this with a compiled C/C++ function and [OpenMP 'atomic update'](https://www.openmp.org/spec-html/5.0/openmpsu95.html).

# In[ ]:


# initial_infections = model.prng.integers(0, 11, size=model.nodes.count, dtype=np.uint32)
prevalence = 0.025  # 2.5% prevalence
initial_infections = np.uint32(np.round(model.prng.poisson(prevalence * initial_populations)))
print(f"Initial infections for first 32 nodes:\n{initial_infections[:32]=}")


@nb.njit((nb.uint32, nb.uint32[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
def initialize_infections(count, infections, nodeid, itimer, inf_mean, inf_std):
    for i in nb.prange(count):
        if infections[nodeid[i]] > 0:
            infections[nodeid[i]] -= 1
            itimer[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))  # must be at least 1 day

    return


# Print this _before_ calling initializing_infections because `initial_infections` is modified (zeroed) in-place.
print(f"{initial_infections.sum()=:,}")

initialize_infections(
    np.uint32(model.population.count),
    initial_infections,
    model.population.nodeid,
    model.population.itimer,
    model.params.inf_mean,
    model.params.inf_std,
)

print(f"{(model.population.itimer > 0).sum()=:,}")


# # Transmission
#
# ## Transmission Part I - Setup
#
# We will add a `network` property to the model to hold the connection weights between the nodes.
#
# We initialize $n_{ij} = n_{ji} = k \frac {P_i^a \cdot P_j^b} {D_{ij}^c}$ where $P_i = \frac {\text{population}_i} N$ and $P_j = \frac {\text{population}_j} N$ respectively.
#
# Then we limit outgoing migration from any one node to `max_frac`.

# In[ ]:


# We need to calculate the distances between the centroids of the nodes in northern Nigeria
model.nodes.add_vector_property("network", model.nodes.count, dtype=np.float32)
network = model.nodes.network

RE = 6371.0  # Earth radius in km


def calc_distance(lat1, lon1, lat2, lon2):
    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d = RE * c
    return d


locations = np.zeros((model.nodes.count, 2), dtype=np.float32)
for i, node in enumerate(nn_nodes.values()):
    (longitude, latitude) = node[1]
    locations[i, 0] = latitude
    locations[i, 1] = longitude
locations = np.radians(locations)

# TODO: Consider keeping the distances and periodically recalculating the network values as the populations change
a = model.params.a
b = model.params.b
c = model.params.c
k = model.params.k
# for i in tqdm(range(model.nodes.count)):
#     popi = initial_populations[i]
#     for j in range(i+1, model.nodes.count):
#         popj = initial_populations[j]
#         network[i,j] = network[j,i] = k * (popi**a) * (popj**b) / (calc_distance(*locations[i], *locations[j])**c)
# network /= np.power(initial_populations.sum(), 2)    # normalize by total population^2

# print(f"Upper left corner of network looks like this (before limiting to max_frac):\n{network[:4,:4]}")

max_frac = model.params.max_frac
# for row in range(network.shape[0]):
#     if (maximum := network[row].sum()) > max_frac:
#         network[row] *= max_frac / maximum

from idmlaser.migration import gravity

# We need to calculate the distances between the centroids of the nodes in northern Nigeria
for i in tqdm(range(model.nodes.count)):
    for j in range(i + 1, model.nodes.count):
        dist = calc_distance(*locations[i], *locations[j])
        network[i, j] = dist
        network[j, i] = dist

model.nodes.network = gravity({"a": a, "b": b, "c": c, "k": k, "max_frac": max_frac}, initial_populations, network)

print(f"Upper left corner of network looks like this (after limiting to max_frac):\n{model.nodes.network[:4,:4]}")


# ## Transmission Part II - Tick/Step Processing Phase
#
# On a tick we accumulate the contagion in each node - currently 1 unit per infectious agent - with `np.add.at()` ([documentation](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)).
#
# We calculate the incoming and outgoing contagion by multiplying by the network connection values.
#
# We determine the force of infection per agent in each node by multiplying by the seasonally adjusted $\beta$ and normalizing by the node population. $\beta_{eff}$ is currently a scalar but it would be trivial to add a `beta` property to `model.nodes` and have per node betas.
#
# We then visit each susceptible agent and draw against the force of infection to determine transmission and draw for duration of infection if transmission occurs.
#
# We will also track incidence by node and tick.

# In[ ]:


@nb.njit(
    (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
    parallel=True,
    nogil=True,
    cache=True,
)
def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence):
    for i in nb.prange(count):
        susceptibility = susceptibilities[i]
        if susceptibility > 0:
            nodeid = nodeids[i]
            force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
            if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                susceptibilities[i] = 0.0  # set susceptibility to 0.0
                # set exposure timer for newly infected individuals to a draw from a normal distribution, must be at least 1 day
                etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(exp_mean, exp_std))))
                incidence[nodeid] += 1

    return


model.nodes.add_vector_property("cases", model.params.ticks, dtype=np.uint32)
model.nodes.add_scalar_property("forces", dtype=np.float32)
model.nodes.add_vector_property("incidence", model.params.ticks, dtype=np.uint32)


def do_transmission_update(model, tick) -> None:
    nodes = model.nodes
    population = model.population

    contagion = nodes.cases[:, tick]  # we will accumulate current infections into this array
    nodeids = population.nodeid[: population.count]  # just look at the active agent indices
    itimers = population.itimer[: population.count]  # just look at the active agent indices
    np.add.at(contagion, nodeids[itimers > 0], 1)  # increment by the number of active agents with non-zero itimer

    network = nodes.network
    transfer = (contagion * network).round().astype(np.uint32)
    contagion += transfer.sum(axis=1)  # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

    forces = nodes.forces
    beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(2 * np.pi * (tick - model.params.seasonality_phase) / 365)
    np.multiply(contagion, beta_effective, out=forces)
    np.divide(forces, model.nodes.population[:, tick], out=forces)  # per agent force of infection

    tx_inner(
        population.susceptibility,
        population.nodeid,
        forces,
        population.etimer,
        population.count,
        model.params.exp_mean,
        model.params.exp_std,
        model.nodes.incidence[:, tick],
    )

    return


# ## Interventions : Serosurveys and SIAs
#
# Let's try an SIA 10 days into the simulation.
#
# _Consider sorting the SIAs by tick in order to be able to just check the first N campaigns in the list._
#

# In[ ]:


from collections import namedtuple

SEROSURVEY = namedtuple("SEROSURVEY", ["tick", "nodes", "age_days_min", "age_days_max"])
SIA = namedtuple("SIA", ["tick", "nodes", "coverage", "age_days_min", "age_days_max"])
EDUCATION = namedtuple("EDUCATION", ["tick", "nodes"])

NINE_MONTHS = 274  # 9 months in days (30 day months + 9/12 of the 5 addition days in a year)
SIX_YEARS = 6 * 365  # 6 years in days

todo = [
    SEROSURVEY(9, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 9, nodes 0-5, [2-6) years old
    SIA(10, [1, 3, 5], 0.80, NINE_MONTHS, SIX_YEARS),  # Tick 10, nodes 1, 3, and 5, 80% coverage, [2-6) years old
    SEROSURVEY(11, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 11, nodes 0-5, [2-6) years old
    EDUCATION(30, [0, 1, 2, 3, 4, 5]),  # Tick 30, nodes 0-5
]

model.nodes.add_vector_property("seronegativity", model.params.ticks, dtype=np.uint32)


@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.uint32[:], nb.int32[:], nb.int32, nb.int32, nb.int32), parallel=True)
def _do_serosurvey(count, targets, nodeids, susceptibilities, seronegativity, dobs, age_min, age_max, tick):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            age = tick - dobs[i]
            if (age_min <= age) and (age < age_max):
                if susceptibilities[i] > 0:
                    seronegativity[nodeids[i]] += 1

    return


def invoke_serosurvey(campaign, model, tick):
    print(f"Running serosurvey {campaign=} at tick {tick}")
    targets = np.zeros(model.nodes.count, dtype=np.uint16)
    targets[campaign.nodes] = 1
    _do_serosurvey(
        model.population.count,
        targets,
        model.population.nodeid,
        model.population.susceptibility,
        model.nodes.seronegativity[:, tick],
        model.population.dob,
        np.int32(campaign.age_days_min),
        np.int32(campaign.age_days_max),
        np.int32(tick),
    )

    return


@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.int32[:], nb.int32, nb.int32, nb.int32), parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, coverage, dobs, age_min, age_max, tick):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            age = tick - dobs[i]
            if (age_min <= age) and (age < age_max):
                if susceptibilities[i] > 0:
                    if np.random.random_sample() < coverage:
                        susceptibilities[i] = 0

    return


def invoke_sia(campaign, model, tick):
    print(f"Running SIA {campaign=} at tick {tick}")
    targets = np.zeros(model.nodes.count, dtype=np.uint16)
    targets[campaign.nodes] = 1
    _do_sia(
        model.population.count,
        targets,
        model.population.nodeid,
        model.population.susceptibility,
        np.float32(campaign.coverage),
        model.population.dob,
        np.int32(campaign.age_days_min),
        np.int32(campaign.age_days_max),
        np.int32(tick),
    )

    return


iv_map = {
    SEROSURVEY: invoke_serosurvey,
    SIA: invoke_sia,
    EDUCATION: lambda campaign, model, tick: print(f"Running education {campaign=} at tick {tick}"),
}


def do_interventions(model, tick):
    while len(todo) > 0 and todo[0][0] == tick:
        campaign = todo.pop(0)
        iv_map[type(campaign)](campaign, model, tick)


for iv in todo:
    if type(iv) not in iv_map:
        raise ValueError(f"Missing invoke function for intervention type: {type(iv).__name__}")


# # **Running the Simulation**
#

# ## Tick/Step Processing Phases
#
# The phases (sub-steps) of the processing on each tick go here as they are implemented.

# In[ ]:


# consider `step_functions` rather than `phases` for the following
model.phases = [
    propagate_population,
    do_births,  # type: ignore
    do_eula_deaths,  # type: ignore
    do_non_disease_deaths,  # type: ignore
    do_infection_update,  # type: ignore
    do_exposure_update,  # type: ignore
    do_transmission_update,  # type: ignore
    do_ri,  # type: ignore
    do_susceptibility_decay,  # type: ignore
    do_interventions,  # type: ignore
]


# ## Iterating Over the Simulation Duration
#
# We iterate over the specified number of ticks, keeping track, in `metrics`, of the time spent in each phase at each tick.

# In[ ]:


from tqdm import tqdm

model.metrics = []
for tick in tqdm(range(model.params.ticks)):
    metrics = [tick]
    for phase in model.phases:
        tstart = datetime.now(tz=None)  # noqa: DTZ005
        phase(model, tick)
        tfinish = datetime.now(tz=None)  # noqa: DTZ005
        delta = tfinish - tstart
        metrics.append(delta.seconds * 1_000_000 + delta.microseconds)  # delta is a timedelta object, let's convert it to microseconds
    model.metrics.append(metrics)


# # Post-Simulation Analysis

# ## Final Population
#
# Let's take a quick look at the final population size accounting for births over the course of the simulation. This does _not_ account for non-disease deaths so we are looking at the maximum number of unique agents over the simulation.

# In[ ]:


print(f"{model.population.count=:,} (vs. requested capacity {model.population.capacity=:,})")


# ## Timing Metrics Part I
#
# Let's convert the timing information to a DataFrame and peek at the first few entries.

# In[ ]:


import pandas as pd

metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
metrics.head()


# ## Timing Metrics Part II
#
# Let's take a look at where we spend our processing time.

# In[ ]:


from matplotlib import pyplot as plt

plot_columns = metrics.columns[1:]
sum_columns = metrics[plot_columns].sum()
print(sum_columns)
print("=" * 36)
print(f"Total: {sum_columns.sum():29,} microseconds")

plt.figure(figsize=(8, 8))
plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%")
plt.title("Sum of Each Column")
plt.show()


# ## Validation - Population Over Time
#
# Let's make sure that our population is growing over time by plotting the population for a few nodes.

# In[ ]:


nodes_to_plot = [0, 1, 2, 3]
node_population = model.nodes.population[nodes_to_plot, :]

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range(model.params.ticks + 1), node_population[i, :], label=f"Node {node}")

plt.xlabel("Tick")
plt.ylabel("Population")
plt.title("Node Population Over Time")
plt.legend()
plt.show()


# ## Validation - Births
#
# Let's see if our births over time look right. Given a fixed CBR and a growing population, we should generally have more births later in the simulation.

# In[ ]:


node_births = model.nodes.births[nodes_to_plot, :]

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range((model.params.ticks + 364) // 365), node_births[i, :], label=f"Node {node}")

plt.xlabel("Year")
plt.ylabel("Births")
plt.title("Node Births Over Time")
plt.legend()
plt.show()


# ## Validation - Non-Disease Deaths
#
# Let's see if our non-disease deaths look right over time.

# In[ ]:


node_deaths = model.nodes.deaths[nodes_to_plot, :]

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range((model.params.ticks + 364) // 365), node_deaths[i, :], label=f"Node {node}")

plt.xlabel("Year")
plt.ylabel("Deaths")
plt.title("Node Deaths Over Time")
plt.legend()
plt.show()


# ## Cases Over Time

# In[ ]:


group = 0
size = 16
nodes_to_plot = list(range(size * group, size * (group + 1)))
nodes_to_plot = [0, 1, 2, 3]

window_start = 0
window_end = 180

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range(window_start, window_end), model.nodes.cases[i, window_start:window_end], label=f"Node {node}")

plt.xlabel("Tick")
plt.ylabel("Cases")
plt.title("Node Cases Over Time")
plt.legend()
plt.show()


# ## Incidence Over Time

# In[ ]:


group = 0
size = 16
nodes_to_plot = list(range(size * group, size * (group + 1)))
nodes_to_plot = [0, 1, 2, 3]

window_start = 0
window_end = 180

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range(window_start, window_end), model.nodes.incidence[i, window_start:window_end], label=f"Node {node}")

plt.xlabel("Tick")
plt.ylabel("Incidence")
plt.title("Node Incidence Over Time")
plt.legend()
plt.show()


# In[ ]:


plt.hist(initial_populations)
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.title("Histogram of Initial Populations")
plt.yscale("log")  # Set y-axis to log scale
plt.show()


# ## Serosurvey Results (Checking the SIA)

# In[ ]:


# convert nodes 0-5 and ticks 5-15 of model.nodes.seronegativity to a DataFrame and print its rows

seronegativity = model.nodes.seronegativity[0:6, 0:16]
seronegativity_df = pd.DataFrame(seronegativity.transpose(), columns=[f"Node {i}" for i in range(6)])
print(seronegativity_df[5:16])
print()
print(np.array(seronegativity_df.iloc[11]) / np.array(seronegativity_df.iloc[9]))


# In[ ]:


print(f"{model.nodes.ri_coverages[0:20]=}")
print(f"{model.population.dob[0:32]=}")
print(f"{model.population.dod[0:32]=}")
print(f"{model.nodes.seronegativity[0:6, 9:12].transpose()=}")
print(f"{model.nodes.cases[0:9,0:16].transpose()=}")
print(f"{model.nodes.births[0:9,0:16].transpose()=}")


# # RI Accessibility Validation
#
# Let's find a node with a reasonable coverage (6 looks good with the default PRNG seed).
# We will find all the births for that node over the course of the simulation.
# We will calculate an expected number of MCV1 recipients, MCV2 recipients, and "none" recipients.
# We will plot the actual count of each from the simulation.
#
# **Note:** This isn't quite correct because the infants _in the initial population_ don't currently get any access to RI.

# In[ ]:


probe_node = 6

istart = 0  # initial_populations.sum()
print(f"{istart=:,}")
iend = model.population.count
print(f"{iend=:,}")
node_births = model.population.nodeid[istart:iend] == probe_node
count = node_births.sum()
print(f"{count=:,}")

ri_coverage = model.nodes.ri_coverages[probe_node]
print(f"{ri_coverage=}")
est_mcv1 = model.params.probability_mcv1_take * ri_coverage
est_mcv2 = model.params.probability_mcv2_take * (1.0 - model.params.probability_mcv1_take) * ri_coverage
est_none = 1.0 - est_mcv1 - est_mcv2
print(f"{est_mcv1=:0.4f}, {est_mcv2=:0.4f}, {est_none=:0.4f}")
expected_mcv1 = np.round(est_mcv1 * count)
expected_mcv2 = np.round(est_mcv2 * count)
expected_none = count - expected_mcv1 - expected_mcv2
print(f"{expected_mcv1=:}, {expected_mcv2=:}, {expected_none=:}")


# In[ ]:


# Assuming 'model.population.accessibility' is a NumPy array or a Pandas Series
accessibility_data = model.population.accessibility[istart:iend][node_births]

# Plot the histogram
counts, bins, patches = plt.hist(accessibility_data, bins=3, edgecolor="black")

# Add labels to each bin
for count, bin_edge in zip(counts, bins):
    plt.text(bin_edge, count, str(int(count)), ha="left", va="bottom")

# Plot the histogram
plt.title("Histogram of Accessibility")
plt.xlabel("Accessibility")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


model.nodes.cases[0:4, 0:4]
