#!/usr/bin/env python

from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Very simple!
class Model:
    pass

model = Model()

# Initialize model nodes and population sizes from data
from mods import init_pop_nigeria as ipn
nn_nodes, initial_populations = ipn.run()


# ## Parameters
# 
# We need some parameters now. We will use `PropertySet` rather than a raw dictionary for the "syntactic sugar" of referencing `params.ticks` rather than `params["ticks"]` each time.
# 
# Also, we will set the parameters separately as `meta_params` and `measles_params` but combine them into one parameter set for future use. We _could_ create `model.params = PropertySet({"meta":meta_params, "measles":measles_params})` and then reference them "by path" in the subsequent code, e.g., `params.meta.ticks` and `params.measles.inf_mean`.


from idmlaser.utils import PropertySet

meta_params = PropertySet({
    "ticks": int(365*10),
    "cbr": 40,  # Nigeria 2015 according to (somewhat random internet source): https://fred.stlouisfed.org/series/SPDYNCBRTINNGA
    "output": Path.cwd() / "outputs",
    "eula_age": 5
})
# parameter?
prevalence = 0.025 # 2.5% prevalence

measles_params = PropertySet({
    "exp_mean": np.float32(7.0),
    "exp_std": np.float32(1.0),
    "inf_mean": np.float32(7.0),
    "inf_std": np.float32(1.0),
    "r_naught": np.float32(14.0),
    "seasonality_factor": np.float32(0.125),
    "seasonality_phase": np.float32(182),
    "ri_coverage": np.float32(0.75),
    "beta_env": np.float32(0.1), # beta(j,0) -- The baseline rate of environment-to-human transmission (all destinations)
    "kappa": np.float32(5e5), # The concentration (number of cells per mL) of V. cholerae required for a 50% probability of infection.
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


# ## Capacity Calculation
# 
# We have our initial populations, but we need to allocate enough space to handle growth during the simulation.

from idmlaser.numpynumba import Population

# We're going to create the human/agent population from the capacity (expansion slots based on births)
# It will be in model.population
Population.create_from_capacity(model,initial_populations)
capacity = model.population.capacity
from schema import schema
model.population.add_properties_from_schema( schema )

# we now have our population dataframe!
# let's give it some values.

# ## Node IDs
# 
# Add a property for node id. 419 nodes requires 9 bits so we will allocate a 16 bit value. Negative IDs don't make sense, so, `uint16`.

# In[5]:


def assign_node_ids(model,initial_populations):
    index = 0
    for nodeid, count in enumerate(initial_populations):
        model.population.nodeid[index:index+count] = nodeid
        index += count

assign_node_ids(model,initial_populations)

# ## Node Populations
# 
# We will need the most recent population numbers in order to determine the births, based on CBR, for the upcoming year. We will also, later, use the current population to determine the effective force of infection, i.e., total contagion / node population.
# 
# Default data type is uint32.

# In[6]:

def save_pops_in_nodes( model, nn_nodes, initial_populations):
    node_count = len(nn_nodes)
    nodes = Population(capacity=node_count) # TODO: rename `Population` to some appropriate to agents _and_ nodes
    model.nodes = nodes # type: ignore
    ifirst, ilast = nodes.add(node_count)
    print(f"{ifirst=:,}, {ilast=:,}")
    model.nodes.add_vector_property("population", model.params.ticks + 1) # type: ignore
    nodes.population[:,0] = initial_populations
    model.nodes.nn_nodes = nn_nodes

save_pops_in_nodes( model, nn_nodes, initial_populations )

# Some of these are inputs and some are outputs
# static inputs
model.nodes.add_vector_property("network", model.nodes.count, dtype=np.float32)
# The climatically driven environmental suitability of V. cholerae by node and time
model.nodes.add_vector_property("psi", model.params.ticks, dtype=np.float32)
model.nodes.psi[:] = 0.001 # placeholder, probably load from csv
# theta: The proportion of the population that have adequate Water, Sanitation and Hygiene (WASH).
model.nodes.add_scalar_property("WASH_fraction", dtype=np.float32) # leave at 0 for now, not used yet

# report outputs
model.nodes.add_vector_property("cases", model.params.ticks, dtype=np.uint32)
model.nodes.add_vector_property("incidence", model.params.ticks, dtype=np.uint32)
model.nodes.add_vector_property("births", (model.params.ticks + 364) // 365)    # births per year
model.nodes.add_vector_property("deaths", (model.params.ticks + 364) // 365)    # deaths per year

# transient for calculations
model.nodes.add_scalar_property("forces", dtype=np.float32)
model.nodes.add_scalar_property("enviro_contagion", dtype=np.float32)

# Add new property "ri_coverages", just randomly for demonstration purposes
# Replace with values from data
model.nodes.add_scalar_property("ri_coverages", dtype=np.float32)
model.nodes.ri_coverages = np.random.rand(len(nn_nodes))
# ri coverages and init prev seem to be the same "kind of thing"?
model.nodes.initial_infections = np.uint32(np.round(np.random.poisson(prevalence*initial_populations)))


# ## Population per Tick
# 
# We will propagate the current populations forward on each tick. Vital dynamics of births and non-disease deaths will update the current values. The argument signature for per tick step phases is (`model`, `tick`). This lets functions access model specific properties and use the current tick, if necessary, e.g. record information or decide to act.

def propagate_population(model, tick):
    model.nodes.population[:,tick+1] = model.nodes.population[:,tick]

    return

from mods import age_init
age_init.init( model )

from mods import mortality
mortality.init( model )

from mods import immunity
immunity.init(model)

# Initial Prevalence
# Print this _before_ initializing infections because `initial_infections` is modified in-place.
#print(f"{(model.population.itimer > 0).sum()=:,}")

from mods import init_prev
# makes reference to specific properties
init_prev.init( model )
#print(f"{initial_infections.sum()=:,}")

# Transmission
from mods import transmission
transmission.init( model )

from mods import intrahost
from mods import maternal_immunity as mi
from mods import ri
from mods import sia
sia.init( model )
from mods import fertility

# ## Tick/Step Processing Phases
# 
# The phases (sub-steps) of the processing on each tick go here as they are implemented.

# In[24]:


# consider `step_functions` rather than `phases` for the following
model.phases = [
    propagate_population,
    fertility.do_births, # type: ignore
    mortality.do_non_disease_deaths, # type: ignore
    intrahost.do_infection_update, # type: ignore
    intrahost.do_exposure_update, # type: ignore
    transmission.do_transmission_update, # type: ignore
    ri.do_ri, # type: ignore
    mi.do_susceptibility_decay, # type: ignore
    sia.do_interventions, # type: ignore 
]


# ## Running the Simulation
# 
# We iterate over the specified number of ticks, keeping track, in `metrics`, of the time spent in each phase at each tick.

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


# ## Final Population
# 
# Let's take a quick look at the final population size accounting for births over the course of the simulation. This does _not_ account for non-disease deaths so we are looking at the maximum number of unique agents over the simulation.

print(f"{model.population.count=:,} (vs. requested capacity {model.population.capacity=:,})")

# ## Timing Metrics Part I
# 
# Let's convert the timing information to a DataFrame and peek at the first few entries.

import final_reports
final_reports.report( model, initial_populations )

