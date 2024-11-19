#!/usr/bin/env python

from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pdb

# Very simple!
class Model:
    pass

model = Model()

# Initialize model nodes and population sizes from data
import importlib.util
import os

# ## Parameters
# 
# We need some parameters now. We will use `PropertySet` rather than a raw dictionary for the "syntactic sugar" of referencing `params.ticks` rather than `params["ticks"]` each time.
# 
# Also, we will set the parameters separately as `meta_params` and `measles_params` but combine them into one parameter set for future use. We _could_ create `model.params = PropertySet({"meta":meta_params, "measles":measles_params})` and then reference them "by path" in the subsequent code, e.g., `params.meta.ticks` and `params.measles.inf_mean`.


from laser_core.propertyset import PropertySet
from idmlaser_cholera.utils import viz_2D, viz_pop, combine_pdfs

meta_params = PropertySet({
    "ticks": int(365*5),
    #"cbr": 40,  # Nigeria 2015 according to (somewhat random internet source): https://fred.stlouisfed.org/series/SPDYNCBRTINNGA
    "output": Path.cwd() / "outputs",
    #"eula_age": 5
})
# parameter?
prevalence = 0.0025 # 2.5% prevalence

measles_params = PropertySet({
    "exp_mean": np.float32(2.0),
    "exp_std": np.float32(1.0),
    "inf_mean": np.float32(6.0),
    "inf_std": np.float32(2.0),
    #"r_naught": np.float32(14.0),
    "r_naught": np.float32(7.0),
    "seasonality_factor": np.float32(0.125),
    "seasonality_phase": np.float32(182),
    "ri_coverage": np.float32(0.75),
    "beta_env": np.float32(1.0), # beta(j,0) -- The baseline rate of environment-to-human transmission (all destinations)
    "kappa": np.float32(5e5), # The concentration (number of cells per mL) of V. cholerae required for a 50% probability of infection.
    "zeta": np.float32(1.0), # Rate that infected individuals shed V. cholerae into the environment. 0.75 is about minimum that gets enviro-only tx.
    "delta_min": np.float32(1/2),
    "delta_max": np.float32(1/60)
})

network_params = PropertySet({
    "a": np.float32(1.0),   # population 1 power factor
    "b": np.float32(1.0),   # population 2 power factor
    "c": np.float32(2.0),   # distance power factor
    "k": np.float32(137.0), # gravity constant
    "max_frac": np.float32(0.5), # maximum fraction of population that can move in a single tick
})

model.params = PropertySet(meta_params, measles_params, network_params) # type: ignore
model.params.beta = 0.4 # model.params.r_naught / model.params.inf_mean # type: ignore

import argparse

# Set up argparse to parse the input directory
parser = argparse.ArgumentParser(description="Specify the input directory.")
parser.add_argument(
    "input_dir",
    nargs="?",
    default=".",
    help="Path to the input directory (default is current directory)"
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="Enable visualization (default is OFF)"
)
parser.add_argument(
    "--to_pdf",
    action="store_true",
    help="Works with --viz. Write to pdf instead of displaying to screen."
)

# Parse the arguments
args = parser.parse_args()

# Assign the input directory to model.params.input_dir
model.params.input_dir = args.input_dir
model.params.viz = args.viz  # This will be True if --viz is specified, False otherwise
model.params.to_pdf = args.to_pdf  # This will be True if --viz is specified, False otherwise

print(f"Input directory set to: {model.params.input_dir}")

# Build the path to the manifest file
manifest_path = os.path.join(model.params.input_dir, "manifest.py")

# Check if manifest.py exists in the specified directory
if os.path.isfile(manifest_path):
    # Load the manifest module
    spec = importlib.util.spec_from_file_location("manifest", manifest_path)
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    print("Manifest module loaded successfully.")
else:
    print(f"Error: {manifest_path} does not exist.")

nn_nodes, initial_populations, cbrs = manifest.load_population_data()

#from laser_core.laserframe import LaserFrame as Population
from idmlaser_cholera.numpynumba.population import ExtendedLF as Population

# We're going to create the human/agent population from the capacity (expansion slots based on births)
# It will be in model.population
from idmlaser_cholera.mods import ages
from idmlaser_cholera.mods import mortality
from idmlaser_cholera.mods import immunity
from idmlaser_cholera.mods import init_prev
from idmlaser_cholera.mods import transmission
from idmlaser_cholera.mods import intrahost
from idmlaser_cholera.mods import maternal_immunity as mi
from idmlaser_cholera.mods import ri
from idmlaser_cholera.mods import sia
from idmlaser_cholera.mods import fertility
from idmlaser_cholera.mods import age_init


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
    #nodes.population[:,0] = initial_populations
    nodes.population[0] = initial_populations
    model.nodes.nn_nodes = nn_nodes


# ## Population per Tick
# 
# We will propagate the current populations forward on each tick. Vital dynamics of births and non-disease deaths will update the current values. The argument signature for per tick step phases is (`model`, `tick`). This lets functions access model specific properties and use the current tick, if necessary, e.g. record information or decide to act.

def propagate_population(model, tick):
    model.nodes.population[tick+1] = model.nodes.population[tick]

    return

def init_from_data():
    Population.create_from_capacity(model,initial_populations,cbrs)
    capacity = model.population.capacity
    from .schema import schema

    model.population.add_properties_from_schema( schema )

    def assign_node_ids(model,initial_populations):
        index = 0
        for nodeid, count in enumerate(initial_populations):
            model.population.nodeid[index:index+count] = nodeid
            index += count
    assign_node_ids(model,initial_populations)

    save_pops_in_nodes( model, nn_nodes, initial_populations )

    # ri coverages and init prev seem to be the same "kind of thing"?
    model.nodes.initial_infections = np.uint32(np.round(np.random.poisson(prevalence*initial_populations)))

    # this is erroring because population size isn't right
    age_init.init( model, manifest )
    immunity.init(model)
    #init_prev.init( model )

    return capacity

def init_from_file( filename ):
    population = Population.load( filename )
    model.population = population

    def extend_capacity_after_loading( model ):
        capacity = model.population.capacity
        print(f"Allocating capacity for {capacity:,} individuals")
        model.population.set_capacity( capacity )

        ifirst, ilast = model.population.current()
        print(f"{ifirst=:,}, {ilast=:,}")
        return capacity
    extend_capacity_after_loading( model )

    save_pops_in_nodes( model, nn_nodes, initial_populations )

from idmlaser_cholera.numpynumba.population import check_hdf5_attributes
from idmlaser_cholera.demographics import cumulative_deaths

def check_for_cached():
    hdf5_directory = manifest.laser_cache
    import os
    if not os.path.exists(hdf5_directory):
        os.makedirs(hdf5_directory)
    for filename in os.listdir(hdf5_directory):
        if filename.endswith(".h5"):
            hdf5_filepath = os.path.join(hdf5_directory, filename)
            cached = check_hdf5_attributes(
                hdf5_filename=hdf5_filepath,
                initial_populations=initial_populations,
                age_distribution=age_init.age_distribution,
                cumulative_deaths=cumulative_deaths
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

ages.init( model )
mortality.init( model )
sia.init( model, manifest )

model.nodes.add_vector_property("cbrs", model.nodes.count, dtype=np.float32)
model.nodes.cbrs = np.array(list(cbrs.values()))
model.nodes.add_vector_property("network", model.nodes.count, dtype=np.float32)
# The climatically driven environmental suitability of V. cholerae by node and time
model.nodes.add_vector_property("psi", model.params.ticks, dtype=np.float32)

# theta: The proportion of the population that have adequate Water, Sanitation and Hygiene (WASH).
model.nodes.add_vector_property("WASH_fraction", model.params.ticks, dtype=np.float32) # leave at 0 for now, not used yet

if model.params.viz:
    viz_pop( model )

transmission.init( model, manifest )

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

# combine pdfs into single
if model.params.to_pdf:
    combine_pdfs()


# consider `step_functions` rather than `phases` for the following
model.phases = [
    propagate_population,
    fertility.step, # type: ignore
    ages.step,
    mortality.step, # type: ignore
    intrahost.step, # type: ignore
    #intrahost.step2, # type: ignore
    transmission.step, # type: ignore
    #ri.step, # type: ignore
    mi.step, # type: ignore
    #sia.step, # type: ignore 
]


# ## Running the Simulation
# 
# We iterate over the specified number of ticks, keeping track, in `metrics`, of the time spent in each phase at each tick.

model.metrics = []
for tick in tqdm(range(model.params.ticks)):
    """
    if tick == 365:
        model.population.save( filename="laser_cache/burnin_cholera.h5", initial_populations=initial_populations, age_distribution=age_init.age_distribution, cumulative_deaths=cumulative_deaths)

    """
    #"""
    if tick == 40:
        init_prev.init( model )
    #"""
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

from . import final_reports
final_reports.report( model, initial_populations )

