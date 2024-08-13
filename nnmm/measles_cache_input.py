#!/usr/bin/env python
# coding: utf-8

# # Model
# 
# We will create a blank object, called `model`, to hold the various pieces of information we have about our model.

# In[540]:

import sys
eula_age = int(sys.argv[1])

class Model:
    pass

model = Model()


total_births = 0

# ## Source Demographics Data
# 
# We have some census data for Nigeria (2015) which we will use to set the initial populations for our nodes. Since we are modeling northern Nigeria, as a first cut we will only choose administrative districts which start with "NORTH_". That gives us 419 nodes with a total population of ~96M.

# In[541]:


# setup initial populations
from pathlib import Path

import numpy as np
import numba as nb
from datetime import datetime
from tqdm import tqdm
import pdb


# ## Parameters
# 
# We need some parameters now. We will use `PropertySet` rather than a raw dictionary for the "syntactic sugar" of referencing `params.ticks` rather than `params["ticks"]` each time.
# 
# Also, we will set the parameters separately as `meta_params` and `measles_params` but combine them into one parameter set for future use. We _could_ create `model.params = PropertySet({"meta":meta_params, "measles":measles_params})` and then reference them "by path" in the subsequent code, e.g., `params.meta.ticks` and `params.measles.inf_mean`.

# In[542]:


from idmlaser.utils import PropertySet

meta_params = PropertySet({
    "ticks": int(3650/2),
    "cbr": 40,  # Nigeria 2015 according to (somewhat random internet source): https://fred.stlouisfed.org/series/SPDYNCBRTINNGA
    "output": Path.cwd() / "outputs",
    "eula_age": eula_age
})

measles_params = PropertySet({
    "exp_mean": np.float32(7.0),
    "exp_std": np.float32(1.0),
    "inf_mean": np.float32(7.0),
    "inf_std": np.float32(1.0),
    "r_naught": np.float32(14.0),
    "seasonality_factor": np.float32(0.125),
    "seasonality_phase": np.float32(182),
    "ri_coverage": np.float32(0.75),
})

network_params = PropertySet({
    "a": np.float32(1.0),   # population 1 power factor
    "b": np.float32(1.0),   # population 2 power factor
    "c": np.float32(2.0),   # distance power factor
    "k": np.float32(137.0), # gravity constant
    "max_frac": np.float32(0.5), # maximum fraction of population that can move in a single tick
})

model.params = PropertySet(meta_params, measles_params, network_params) # type: ignore
model.params.beta = model.params.r_naught / model.params.inf_mean # type: ignore


from nigeria import lgas

print(Path.cwd())
admin2 = {k:v for k,v in lgas.items() if len(k.split(":")) == 5}
print(f"{len(admin2)=}")

nn_nodes = {k:v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
print(f"{len(nn_nodes)=}")

initial_populations = np.array([v[0][0] for v in nn_nodes.values()])
print(f"{len(initial_populations)=}")
print(f"First 32 populations:\n{initial_populations[0:32]}")
print(f"{initial_populations.sum()=:,}")


# ## Capacity Calculation
# 
# We have our initial populations, but we need to allocate enough space to handle growth during the simulation.

# In[543]:


from idmlaser.numpynumba import Population
def extend_capacity(model, initial_populations):
    capacity = initial_populations.sum()
    print(f"initial {capacity=:,}")
    print(f"{model.params.cbr=}, {model.params.ticks=}")    # type: ignore
    growth = ((1.0 + model.params.cbr/1000)**(model.params.ticks // 365))   # type: ignore
    print(f"{growth=}")
    capacity *= growth
    capacity *= 1.01  # 1% buffer
    capacity = np.uint32(np.round(capacity))
    print(f"required {capacity=:,}")
    print(f"Allocating capacity for {capacity:,} individuals")
    population = Population(capacity)
    model.population = population   # type: ignore
    ifirst, ilast = population.add(initial_populations.sum())
    print(f"{ifirst=:,}, {ilast=:,}")

def extend_capacity_after_loading( model ):
    capacity = model.population.capacity
    print(f"Allocating capacity for {capacity:,} individuals")
    model.population.set_capacity( capacity )

    ifirst, ilast = model.population.current()
    print(f"{ifirst=:,}, {ilast=:,}")
    return capacity


def add_node_property( population, initial_populations ):
    population.add_scalar_property("nodeid", np.uint16)
    index = 0
    for nodeid, count in enumerate(initial_populations):
        population.nodeid[index:index+count] = nodeid
        index += count

# initialize susceptibility based on age
@nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True)
def initialize_susceptibility(count, dob, susceptibility):
    for i in nb.prange(count):
        if dob[i] >= -365*5:
            susceptibility[i] = 1

    return
def initialize_immunity():
    # Initialize everyone with Accessibility 'IP'. Values are 0 (High), 1 (Medium) or 2 (Low), with probability 0.85, 0.14, 0.01 conditioned on node-based coverage.
    model.population.add_scalar_property("accessibility", np.uint8)
    model.population.add_scalar_property("susceptibility", np.uint8)
    model.population.add_scalar_property("susceptibility_timer", np.uint8) # might be too small, currently thinking 6 months
    model.population.add_scalar_property("ri_timer", np.uint16)
    initialize_susceptibility(model.population.count, model.population.dob, model.population.susceptibility)


# ## Non-Disease Mortality Part I
# 
# We start by loading a population pyramid in order to initialize the ages of the initial population realistically.
# 
# The population pyramid is typically in 5 year age buckets. Once we draw for the age bucket, we draw uniformly for a date of birth within the range of the bucket.
# 
# **Note:** the values in `model.population.dob` are _positive_ at this point. Later we will negate them to convert them to dates of birth prior to now (t = 0).

# In[549]:

from tqdm import tqdm
import idmlaser.pyramid as pyramid
pyramid_file = Path.cwd().parent / "tests" / "USA-pyramid-2023.csv"
age_distribution = pyramid.load_pyramid_csv(pyramid_file)

def add_dobs(population, initial_populations):

    print(f"Loading pyramid from '{pyramid_file}'...")
    print("Creating aliased distribution...")
    aliased_distribution = pyramid.AliasedDistribution(age_distribution[:,4])
    count_active = initial_populations.sum()
    print(f"Sampling {count_active:,} ages... {population.count=:,}")
    buckets = aliased_distribution.sample(population.count)
    minimum_age = age_distribution[:, 0] * 365      # closed, include this value
    limit_age = (age_distribution[:, 1] + 1) * 365  # open, exclude this value
    population.add_scalar_property("dob", np.int32)
    mask = np.zeros(population.capacity, dtype=bool)

    print("Converting age buckets to ages...")
    for i in tqdm(range(len(age_distribution))):
        mask[:count_active] = (buckets == i)    # indices of agents in this age group bucket
        # draw uniformly between the start and end of the age group bucket
        population.dob[mask] = np.random.randint(low=minimum_age[i], high=limit_age[i], size=mask.sum())


# ## Non-Disease Mortality Part II
# 
# We import `pdsod` ("predicted_days_of_death") which currently uses a USA 2003 survival table. `pdsod` will draw for a year of death (in or after the current age year) and then will draw a day in the selected year.
# 
# **Note:** the incoming values in `model.population.dob` are positive (_ages_). After we use them to draw for date of death, we negate them to convert them to dates of birth (in days) prior to now (t=0).

# In[550]:


def add_dods(population):
    from idmlaser.kmcurve import pdsod  # noqa: F811

    population.add_scalar_property("dod", np.int32)
    dobs = population.dob
    dods = population.dod
    tstart = datetime.now(tz=None)  # noqa: DTZ005
    dods[0:population.count] = pdsod(dobs[0:population.count], max_year=100)
    tfinish = datetime.now(tz=None)  # noqa: DTZ005
    print(f"Elapsed time for drawing dates of death: {tfinish - tstart}")

    dods -= dobs.astype(dods.dtype) # renormalize to be relative to _now_ (t = 0)
    dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative
    print(f"First 32 DoBs (should all be negative - these agents were born before today):\n{dobs[:32]}")
    print(f"First 32 DoDs (should all be positive - these agents will all pass in the future):\n{dods[:32]}")
    model.population.init_eula(model.params.eula_age)

# ## Initial Infections
# 
# We choose up to 10 infections per node initially. Another option would be to choose the initial number of infections based on estimated prevalence.
# 
# **Note:** this function has a known bug based on using Numba to optimize the function. [Numba only exposes an atomic decrement in CUDA implementations](https://numba.readthedocs.io/en/stable/cuda/intrinsics.html) so on the CPU it is possible that more than one thread decides there are still agents to initialize as infected and the total count of initial infections is more than specified.
# 
# **_TODO:_** After checking to see if there are still infections to be initialized in this node, consider checking the susceptibility of the agent as well, i.e., only infect susceptible agents.
# 
# **_TODO_:** Fix this with a compiled C/C++ function and [OpenMP 'atomic update'](https://www.openmp.org/spec-html/5.0/openmpsu95.html).

# In[556]:


# initial_infections = np.random.randint(0, 11, model.nodes.count, dtype=np.uint32)
prevalence = 0.025 # 2.5% prevalence

@nb.njit((nb.uint32, nb.uint32[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
def initialize_infections(count, infections, nodeid, itimer, inf_mean, inf_std):

    for i in nb.prange(count):
        if infections[nodeid[i]] > 0:
            infections[nodeid[i]] -= 1
            itimer[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std)))) # must be at least 1 day

    return


def init_infections(initial_populations):
    # Print this _before_ initializing infections because `initial_infections` is modified in-place.
    #initial_infections = np.uint32(np.round(np.random.poisson(prevalence*model.nodes.population[:,0])))
    initial_infections = np.uint32(np.round(np.random.poisson(prevalence*initial_populations)))
    print(f"{initial_infections.sum()=:,}")

    initialize_infections(np.uint32(model.population.count), initial_infections, model.population.nodeid, model.population.itimer, model.params.inf_mean, model.params.inf_std)

    print(f"{(model.population.itimer > 0).sum()=:,}")


def init_from_data():
    #nn_nodes, initial_populations = init_pop_from_data()
    capacity = extend_capacity( model, initial_populations )
    add_node_property( model.population, initial_populations )
    add_dobs( model.population, initial_populations )
    add_dods( model.population )
    model.population.add_scalar_property("etimer", np.uint8)
    model.population.add_scalar_property("itimer", np.uint8)
    init_infections(initial_populations)
    initialize_immunity()
    return capacity

def init_from_file( filename ):
    population = Population.load( filename )
    model.population = population
    extend_capacity_after_loading( model )

from idmlaser.kmcurve import cumulative_deaths
from idmlaser.numpynumba.population import check_hdf5_attributes

def check_for_cached():
    hdf5_directory = "laser_cache"
    #pdb.set_trace()
    import os
    if not os.path.exists(hdf5_directory):
        os.makedirs(hdf5_directory)
    for filename in os.listdir(hdf5_directory):
        if filename.endswith(".h5"):
            hdf5_filepath = os.path.join(hdf5_directory, filename)
            cached = check_hdf5_attributes(
                hdf5_filename=hdf5_filepath,
                initial_populations=initial_populations,
                age_distribution=age_distribution,
                cumulative_deaths=cumulative_deaths,
                eula_age=model.params.eula_age
            )
            if cached:
                init_from_file( hdf5_filepath )
                return True
    return False

not_cached = True
if check_for_cached():
    print( "*\nFound cached file. Using it.\n*" )
    not_cached = False # sorry for double negative
else:
    capacity = init_from_data()

# ## Node IDs
# 
# Add a property for node id. 419 nodes requires 9 bits so we will allocate a 16 bit value. Negative IDs don't make sense, so, `uint16`.


# ## Node Populations
# 
# We will need the most recent population numbers in order to determine the births, based on CBR, for the upcoming year. We will also, later, use the current population to determine the effective force of infection, i.e., total contagion / node population.
# 
# Default data type is uint32.

# In[545]:


from nigeria import lgas
admin2 = {k:v for k,v in lgas.items() if len(k.split(":")) == 5}
nn_nodes = {k:v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}

node_count = len(nn_nodes) # TBD: unify solution
#node_count = model.population.node_count # len(nn_nodes)
nodes = Population(capacity=node_count) # TODO: rename `Population` to some appropriate to agents _and_ nodes
model.nodes = nodes # type: ignore
ifirst, ilast = nodes.add(node_count)
print(f"{ifirst=:,}, {ilast=:,}")
nodes.add_vector_property("population", model.params.ticks + 1) # type: ignore
# maybe only if loaded file... ?
#pdb.set_trace()
#initial_populations = model.population.current_populations()
#print(f"First 32 populations:\n{initial_populations[0:32]}")
nodes.population[:,0] = initial_populations 

# only for reloading from file; add some condition

# # RI Coverages

# In[546]:


# Add new property "ri_coverages"
nodes.add_scalar_property("ri_coverages", dtype=np.float32)
nodes.ri_coverages = np.random.rand(node_count)


# ## Population per Tick
# 
# We will propagate the current populations forward on each tick. Vital dynamics of births and non-disease deaths will update the current values. The argument signature for per tick step phases is (`model`, `tick`). This lets functions access model specific properties and use the current tick, if necessary, e.g. record information or decide to act.

# In[547]:


def propagate_population(model, tick):
    model.nodes.population[:,tick+1] = model.nodes.population[:,tick]

    return



from idmlaser.kmcurve import pdsod

model.nodes.add_vector_property("births", (model.params.ticks + 364) // 365)    # births per year

# Adding ri_timer here since it's referred to in do_births.
import access_groups as ag
import ri
import maternal_immunity as mi

# ## Non-Disease Mortality Part III
# 
# If we had no births, we could use a regular FIFO queue with the dates of death sorted to process non-disease deaths in time order. However, we may have agents born during the simulation who are predicted to die from non-disease causes before the end of the simulation and need to be insert, in sorted order, into the queue. So, we will use a priority queue - a data structure which allows for efficient insertion and removal while maintaining sorted order - to keep track of agents by date of non-disease death.

# In[551]:


def init_nondisease_mortality_queue( model, capacity ):
    from idmlaser.utils import PriorityQueuePy

    tstart = datetime.now(tz=None)  # noqa: DTZ005
    dods = model.population.dod
    mortality = PriorityQueuePy(capacity, dods)
    for i in tqdm(indices := np.nonzero(dods[0:model.population.count] < model.params.ticks)[0]):
        mortality.push(i)
    tfinish = datetime.now(tz=None)  # noqa: DTZ005

    print(f"Elapsed time for pushing dates of death: {tfinish - tstart}")
    print(f"Non-disease mortality: tracked {len(indices):,}, untracked {model.population.count - len(indices):,}")

    model.nddq = mortality

init_nondisease_mortality_queue( model, model.population.capacity )

# ## Non-Disease Mortality Part IV
# 
# During the simulation, we will process all agents with a predicated date of death today marking them as deceased and removing them from the active population count.
# 
# **_TODO:_** Set a state to indicate "deceased" _or_ ignore the priority queue altogether and compare `dod` against `tick` and ignore agents where `dod < tick`.

# In[552]:


model.nodes.add_vector_property("deaths", (model.params.ticks + 364) // 365)    # deaths per year



## Initialization complete, Step functions...

# ## Vital Dynamics: Births
# 
# Let's implement births over time. We will use the CBR in `model.params` and draw for the number of births this year based on the most recent population. Then, we will distribute those births as evenly as possible for integral values over the days of the year.
# 
# Note that we add in the date of birth and date of non-disease death after we add those properties below.
# 
# Note that we add in initializing the susceptibility after we add that property below.

# In[548]:

def do_births(model, tick):
    doy = tick % 365 + 1    # day of year 1...365
    year = tick // 365

    if doy == 1:
        #pdb.set_trace()
        model.nodes.births[:, year] = np.random.poisson(model.nodes.population[:, tick] * model.params.cbr / 1000)

    annual_births = model.nodes.births[:, year]
    todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
    count_births = todays_births.sum()

    global total_births
    total_births += count_births

    istart, iend = model.population.add(count_births)   # add count_births agents to the population, get the indices of the new agents

    # enable this after loading the aliased distribution and dod and dob properties (see cells below)
    model.population.dob[istart:iend] = tick    # now update dob to reflect being born today
    model.population.dod[istart:iend] = pdsod(model.population.dob[istart:iend], max_year=100)   # make use of the fact that dob[istart:iend] is currently 0


    index = istart
    nodeids = model.population.nodeid   # grab this once for efficiency
    dods = model.population.dod # grab this once for efficiency
    max_tick = model.params.ticks
    for nodeid, births in enumerate(todays_births):
        nodeids[index:index+births] = nodeid
        for agent in range(index, index+births):
            # If the agent will die before the end of the simulation, add it to the queue
            if dods[agent] < max_tick:
                model.nddq.push(agent)
        index += births
    model.nodes.population[:,tick+1] += todays_births

    ag.set_accessibility( model, istart, iend )
    # accessibility ust be set before ri, since currently ri uses accessibility
    ri.add_with_ips( model, count_births, istart, iend )
    mi.add( model, istart, iend )
    return

def do_non_disease_deaths(model, tick):

    # Add eula population
    year = tick // 365
    for nodeid in range(len(model.population.total_population_per_year)):
        model.nodes.population[nodeid,tick+1] -= (model.population.expected_new_deaths_per_year[nodeid][year]/365)

    pq = model.nddq
    while len(pq) > 0 and pq.peekv() <= tick:
        i = pq.popi()
        nodeid = model.population.nodeid[i]
        model.nodes.population[nodeid,tick+1] -= 1
        model.nodes.deaths[nodeid,year] += 1

    return



# ## Incubation and Infection
# 
# We will add incubation timer (`etimer` for "exposure timer") and an infection (or infectious) timer (`itimer`) properties to the model population. A `uint8` counting down from as much as 255 days will be more than enough.
# 
# We wrap a Numba compiled function using all available cores in the infection update function, extracting the `count` and `itimer` values the JITted function needs.
# 
# Similarly, we wrap a Numba compiled function using all available cores in the exposure update function, extracting the values the JITted function needs from the `model` object.

# In[553]:


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
                itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))    # must be at least 1 day

    return

def do_exposure_update(model, tick):

    _exposure_update(nb.uint32(model.population.count), model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std)

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

# In[554]:



# ## Routine Immunization - RI
# 
# - Children under 1(?) year of age should draw for a date of RI (consider whether to draw vs. RI coverage in the node before choosing an RI date or later when their RI date comes up). Children with an RI date in the future should be put in a priority queue to be processed on their date of RI.
#   - There can be a single `riq` for the model, there is no need to differentiate by node (any node heterogeneity can be considered when their RI date comes up).
# 
# - Create a tick/step phase function to process all agents with RI due on the given tick (see do_non_disease_deaths). This function draws to determine if the innoculation happens (based on RI coverage if the draw is not done at birth), determines if the innoculation is efficacious (potential maternal immunity interference or just vaccine failure). If the vaccination does not take, the agent should be considered for a follow-up RI at 15 months which would be in an additional RI priority queue.
# 
# - Update do_births to draw for RI (based on coverage) and RI date (based on RI date distribution) and put them into the `model.riq` (or set an RI timer...).

# In[555]:


def do_ri(model, tick):
    import pdb
    ri._update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility, model.population.age_at_vax, model.population.dob, tick)
    return


def do_susceptibility_decay(model, tick):
    mi._update_susceptibility_based_on_sus_timer(model.population.count, model.population.susceptibility_timer, model.population.susceptibility)
    return



# ## Transmission Part I - Setup
# 
# We will add a `network` property to the model to hold the connection weights between the nodes.
# 
# We initialize $n_{ij} = n_{ji} = k \frac {P_i^a \cdot P_j^b} {D_{ij}^c}$
# 
# Then we limit outgoing migration from any one node to `max_frac`.

# In[557]:


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
for i in tqdm(range(model.nodes.count)):
    popi = initial_populations[i]
    for j in range(i+1, model.nodes.count):
        popj = initial_populations[j]
        network[i,j] = network[j,i] = k * (popi**a) * (popj**b) / (calc_distance(*locations[i], *locations[j])**c)
network /= np.power(initial_populations.sum(), c)    # normalize by total population^2

print(f"Upper left corner of network looks like this (before limiting to max_frac):\n{network[:4,:4]}")

max_frac = model.params.max_frac
for row in range(network.shape[0]):
    if (maximum := network[row].sum()) > max_frac:
        network[row] *= max_frac / maximum

print(f"Upper left corner of network looks like this (after limiting to max_frac):\n{network[:4,:4]}")

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

# In[558]:


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
            force = susceptibility * forces[nodeid] # force of infection attenuated by personal susceptibility
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

    contagion = nodes.cases[:, tick]    # we will accumulate current infections into this array
    nodeids = population.nodeid[:population.count]  # just look at the active agent indices
    itimers = population.itimer[:population.count] # just look at the active agent indices
    np.add.at(contagion, nodeids[itimers > 0], 1)   # increment by the number of active agents with non-zero itimer

    network = nodes.network
    transfer = (contagion * network).round().astype(np.uint32)
    contagion += transfer.sum(axis=1)   # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)   # decrement by outgoing "migration"

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


# ## SIAs
# 
# Let's try an SIA 10 days into the simulation.
# 
# _Consider sorting the SIAs by tick in order to be able to just check the first N campaigns in the list._
# 

# In[559]:


sias = [(10, [1, 3, 5], 0.80)]  # Tick 10, nodes 1, 3, and 5, 80% coverage.

@nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32), parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, coverage):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            if susceptibilities[i] > 0:
                if np.random.random_sample() < coverage:
                    susceptibilities[i] = 0
    return

def do_sias(model, tick):
    while len(sias) > 0 and sias[0][0] == tick:
        campaign = sias.pop(0)
        print(f"Running SIA {campaign=} at tick {tick}")
        (day, nodes, coverage) = campaign
        targets = np.zeros(model.nodes.count, dtype=np.uint8)
        targets[nodes] = 1
        _do_sia(model.population.count, targets, model.population.nodeid, model.population.susceptibility, coverage)


# ## Tick/Step Processing Phases
# 
# The phases (sub-steps) of the processing on each tick go here as they are implemented.

# In[560]:


# consider `step_functions` rather than `phases` for the following
model.phases = [
    propagate_population,
    do_births, # type: ignore
    do_non_disease_deaths, # type: ignore
    do_infection_update, # type: ignore
    do_exposure_update, # type: ignore
    do_transmission_update, # type: ignore
    do_ri, # type: ignore
    do_susceptibility_decay, # type: ignore
    do_sias, # type: ignore
]


# ## Running the Simulation
# 
# We iterate over the specified number of ticks, keeping track, in `metrics`, of the time spent in each phase at each tick.

# In[561]:


def save_initial_pop():
    # Need to save some metadata/data that was critical to the creation of the init population
    # initial population is at: model.nodes.population[:,0]
    # age_distribution
    # Generate a unique filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"laser_cache/pop_init_eulagized_{timestamp}.h5"
    #pdb.set_trace()
    model.population.save( filename, initial_populations=initial_populations, age_distribution=age_distribution, cumulative_deaths=cumulative_deaths, eula_age=model.params.eula_age )

if not_cached:
    save_initial_pop()


model.population.add_scalar_property("age_at_vax", np.uint16)
#population.add_scalar_property("age_at_inf", np.uint16)
model.metrics = []
for tick in tqdm(range(model.params.ticks)):
    metrics = [tick]
    for phase in model.phases:
        tstart = datetime.now(tz=None)  # noqa: DTZ005
        phase(model, tick)
        tfinish = datetime.now(tz=None)  # noqa: DTZ005
        delta = tfinish - tstart
        metrics.append(delta.seconds * 1_000_000 + delta.microseconds)  # delta is a timedelta object, let's convert it to microseconds
        #if tick == 1824: # 365*5:
            #model.population.save(filename="pop_5yrs.hd5", tail_number=total_births)
            #model.population.save_npz(filename="pop_5yrs.npz", tail_number=total_births)
    model.metrics.append(metrics)


# ## Final Population
# 
# Let's take a quick look at the final population size accounting for births over the course of the simulation. This does _not_ account for non-disease deaths so we are looking at the maximum number of unique agents over the simulation.

# In[562]:


print(f"{model.population.count=:,} (vs. requested capacity {model.population.capacity=:,})")


# ## Timing Metrics Part I
# 
# Let's convert the timing information to a DataFrame and peek at the first few entries.

# In[563]:


import pandas as pd

metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
metrics.head()


# ## Timing Metrics Part II
# 
# Let's take a look at where we spend our processing time.

# In[564]:


import matplotlib.pyplot as plt

plot_columns = metrics.columns[1:]
sum_columns = metrics[plot_columns].sum()
print(sum_columns)
print("=" * 33)
print(f"Total: {sum_columns.sum():26,}")
plt.figure(figsize=(8, 8))
plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%")
plt.title("Sum of Each Column")
plt.show()


# ## Validation - Population Over Time
# 
# Let's make sure that our population is growing over time by plotting the population for a few nodes.

# In[565]:


from matplotlib import pyplot as plt

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

# In[566]:


from matplotlib import pyplot as plt

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

# In[567]:


from matplotlib import pyplot as plt

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

# In[571]:


from matplotlib import pyplot as plt

group = 0
size = 16
nodes_to_plot = list(range(size*group,size*(group+1)))
nodes_to_plot = [ 0, 1, 2, 3 ]

window_start = 0
window_end = 180

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range(window_start,window_end), model.nodes.cases[i, window_start:window_end], label=f"Node {node}")

plt.xlabel("Tick")
plt.ylabel("Cases")
plt.title("Node Cases Over Time")
plt.legend()
plt.show()


# ## Incidence Over Time

# In[572]:


from matplotlib import pyplot as plt

group = 0
size = 16
nodes_to_plot = list(range(size*group,size*(group+1)))
nodes_to_plot = [ 0, 1, 2, 3 ]

window_start = 0
window_end = 180

plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes_to_plot):
    plt.plot(range(window_start,window_end), model.nodes.incidence[i, window_start:window_end], label=f"Node {node}")

plt.xlabel("Tick")
plt.ylabel("Incidence")
plt.title("Node Incidence Over Time")
plt.legend()
plt.show()


# In[569]:


import matplotlib.pyplot as plt

plt.hist(initial_populations)
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.title('Histogram of Initial Populations')
plt.yscale('log')  # Set y-axis to log scale
plt.show()

