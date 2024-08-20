commit 2ce4764876daa91d69b2212c02d1bec3b5b8ec7e
Author: Christopher Lorton <christopher.lorton@gatesfoundation.org>
Date:   Fri Aug 2 15:45:04 2024 -0700

    Add plain script of FUSION code.

diff --git a/nnmm/measles.py b/nnmm/measles.py
new file mode 100644
index 0000000..3ec43ae
--- /dev/null
++ b/nnmm/measles.py
@@ -0,0 +1,1194 @@
# # **Welcome to LASER FUSION\***
# 
# \***FU**LL **SI**MULATION IN **O**NE **N**OTEBOOK
# 
# This notebook is an exercise in putting _all_ the required functionality for a LASER style model into a single notebook. We will use the results of this exercise to determine what functionality is common and can be extracted to features in the LASER Python package (e.g., demographics functions, spatial connectivity functions, sorted and static queues) and what functionality is specific to our northern Nigeria measles modeling and would be layered over LASER package functionality.

# # Model
# 
# We will create a blank object, called `model`, to hold the various pieces of information we have about our model.


from datetime import datetime

print(f"Running FUSION at {datetime.now()}")
from pathlib import Path

import numpy as np

    "ticks": 3650,
    "prng_seed": 20240801,
ri_params = PropertySet({
    "ri_coverage": np.float32(0.75),
    "mcv1_start": int(8.5*365/12),
    "mcv1_end": int(9.5*365/12),
    "mcv2_start": int(14.5*365/12),
    "mcv2_end": int(15.5*365/12),
    "probability_mcv1_take": np.float32(0.85),
    "probability_mcv2_take": np.float32(0.95),
})

model.params = PropertySet(meta_params, measles_params, network_params, ri_params) # type: ignore
# ## PRNG Seeding
# 
# **_Note 1:_** We will use the `default_rng` from `numpy.random` for the model PRNG and `np.random` for the Numba PRNG (Numba rewrites np.random to its parallelized PRNG).
# 
# **_Note 2:_** PRNG seeding for reproducibility depends on using the same number of cores (or enforced with `nb.set_num_threads()` or setting the `NUMBA_NUM_THREADS` environment variable?).

# In[4]:


import numba as nb

@nb.njit(parallel=True)
def set_numba_seeds(seed: np.uint32):

    for t in nb.prange(nb.get_num_threads()):
        np.random.seed(seed + t)

    return

print(f"Setting the model PRNG and Numba PRNG seeds for {nb.get_num_threads()} threads using seed {model.params.prng_seed}")
model.prng = np.random.default_rng(np.uint32(model.params.prng_seed))
set_numba_seeds(np.uint32(model.prng.integers(0, 2**32)))


# In[5]:
# Add a property for node IDs. 419 nodes requires 9 bits so we will allocate a 16 bit value. Negative IDs don't make sense, so, `uint16`.
# In[6]:
print(f"Sample node IDs: {','.join([str(model.population.nodeid[i]) for i in range(0, initial_populations.sum(), initial_populations.sum() // 32)])}")

# Default data type is `uint32`.
# In[7]:
nodes = Population(capacity=node_count) # TODO: rename `Population` to something appropriate to agents _and_ nodes
# In[8]:

# In[9]:
        model.nodes.births[:, year] = model.prng.poisson(model.nodes.population[:, tick] * model.params.cbr / 1000)
    model.population.dod[istart:iend] = tick + pdsod(model.population.dob[istart:iend], max_year=100, prng=model.prng)   # make use of the fact that dob[istart:iend] is currently 0
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

# 
# # Non-Disease Mortality
# 
# ## Non-Disease Mortality Part I
# In[10]:
aliased_distribution = pyramid.AliasedDistribution(age_distribution[:,4], prng=model.prng)
print(f"Sampling {count_active:,} ages... {model.population.count=:,}")
buckets = aliased_distribution.sample(model.population.count)
print(f"First 32 buckets:\n{buckets[:32]}")
model.population.add_scalar_property("dob", np.int32)
    model.population.dob[mask] = model.prng.integers(low=minimum_age[i], high=limit_age[i], size=mask.sum())

print(f"First 32 ages:\n{model.population.dob[0:32]=}")
# ## Non-Disease Mortality Part II
# In[11]:
dods[0:population.count] = pdsod(dobs[0:population.count], max_year=100, prng=model.prng)
# ## Non-Disease Mortality Part III
# In[12]:
# ## Non-Disease Mortality Part IV
# In[13]:
    nodeids = model.population.nodeid
    node_population = model.nodes.population[:, tick+1]
    node_deaths = model.nodes.deaths[:, year]
    susceptibility = model.population.susceptibility
        nodeid = nodeids[i]
        node_population[nodeid] -= 1
        node_deaths[nodeid] += 1
        susceptibility[i] = 0   # disqualifies from infection etc.
# In[14]:
# In[15]:
# # Routine Immunization (RI)
# 
# Do MCV1 for most kids at around 9 months. Do MCV2 for most of the kids who didn't take around 15 months. There are a bunch of ways of doing this. What we're doing here is using an IP -- to borrow EMOD terminology -- for Accessibility, with values 0 (Easy), 1 (Medium), 2 (Hard). See elsewhere for that. The MCV1 timer is a uniform draw from 8.5 months to 9.5 months. MCV2 timer is a uniform draw from 14.5 months to 15.5 months.*
# 
# *See RI parameters near the top.

# ## Coverages
# 
# In[16]:
model.nodes.add_scalar_property("ri_coverages", dtype=np.float32)
model.nodes.ri_coverages = model.prng.random(node_count)

print(f"RI coverage in first 32 nodes {model.nodes.ri_coverages[0:32]=}")
# 
# 
# Vary as needed (see RI parameters near top).
# In[17]:
GET_MCV1 = 0
GET_MCV2 = 1
GET_NONE = 2


    # determine the accessibilty thresholds _by node_ - could be done once if coverage is constant over time
    mcv1_cutoff = model.nodes.ri_coverages * model.params.probability_mcv1_take # probability of (MCV1 vaccination) _and_ (MCV1 take)
    mcv2_cutoff = mcv1_cutoff + model.nodes.ri_coverages * (1.0 - model.params.probability_mcv1_take) * model.params.probability_mcv2_take # probabilty of (MCV1 vaccinaton) _and_ (not MCV1 take) and (MCV2 take)

    random_values = model.prng.random(size=(iend-istart))

    nodeids = model.population.nodeid[istart:iend]  # get slice of nodeids for the new agents
    get_mcv2 = (random_values > mcv1_cutoff[nodeids]) & (random_values <= mcv2_cutoff[nodeids])
    get_none = (random_values > mcv2_cutoff[nodeids])
    accessibility = model.population.accessibility[istart:iend] # get slice of accessibility for the new agents
    accessibility[get_mcv2] = GET_MCV2
    accessibility[get_none] = GET_NONE
    return
# ## Set RI Timers
# 
# Set newborns' `ri_timer` values vased on their accessibility property.
# In[18]:

model.population.add_scalar_property("ri_timer", np.uint16) # Use uint16 for timer since 15 months = 450 days > 2^8

def add_with_ips(model, istart, iend):
    count = iend - istart
    ri_timer_values_9mo = model.prng.integers(model.params.mcv1_start, model.params.mcv1_end, count).astype(np.uint16)
    ri_timer_values_15mo = model.prng.integers(model.params.mcv2_start, model.params.mcv2_end, count).astype(np.uint16)

    # Get slice of the accessibility values for the current agents
    accessibility = model.population.accessibility[istart:iend]

    # Masks based on accessibility
    mask_mcv1 = (accessibility == GET_MCV1)
    mask_mcv2 = (accessibility == GET_MCV2)
    mask_none = (accessibility == GET_NONE) # for validation
    if mask_mcv2.sum() == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 1 (medium)." )
    if mask_none.sum() == 0:
    # mask_none is unnecessary since we don't apply any timer for it

    # Apply the first RI/MCV1 timer to accessibility GET_MCV1
    timers = model.population.ri_timer[istart:iend] # get slice of the RI timers for the new agents
    timers[mask_mcv1] = ri_timer_values_9mo[mask_mcv1]

    # Apply the second RI/MCV2 timer to accessibility GET_MCV2
    timers[mask_mcv2] = ri_timer_values_15mo[mask_mcv2]

    # No need to apply anything to accessibility GET_NONE, as it should remain unmodified

    return


# ## Update RI Timers on Tick

# In[19]:
# Define the function to decrement ri_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility):
        timer = ri_timer[i]
        if timer > 0:
            timer -= 1
            ri_timer[i] = timer
            if timer == 0:
    _update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility)
# # Maternal Immunity
# 
# All newborns come into the world with susceptibility=0. They get a 6 month timer. When that timer hits 0, they become susceptible.
# 
# **_Note:_** This interacts well with RI - RI too early sets susceptibility to 0, but the waning maternal immunity timer goes off _later_ and (re) sets susceptibility to 1.
# ## Initialize Maternal Immunity for Newborns

# In[20]:
model.population.add_scalar_property("susceptibility_timer", np.uint8)  # 6 months in days ~180 < 2^8
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(6*365/12) # 6 months in days
    return
    return


# # Initial Infections
# In[21]:
# initial_infections = model.prng.integers(0, 11, size=model.nodes.count, dtype=np.uint32)
initial_infections = np.uint32(np.round(model.prng.poisson(prevalence*initial_populations)))
print(f"Initial infections for first 32 nodes:\n{initial_infections[:32]=}")
# Print this _before_ calling initializing_infections because `initial_infections` is modified (zeroed) in-place.
# # Transmission
# 
# We initialize $n_{ij} = n_{ji} = k \frac {P_i^a \cdot P_j^b} {D_{ij}^c}$ where $P_i = \frac {\text{population}_i} N$ and $P_j = \frac {\text{population}_j} N$ respectively.
# In[22]:
network /= np.power(initial_populations.sum(), 2)    # normalize by total population^2
# In[23]:
# ## Interventions : Serosurveys and SIAs
# In[24]:


from collections import namedtuple
SEROSURVEY = namedtuple("SEROSURVEY", ["tick", "nodes", "age_days_min", "age_days_max"])
SIA = namedtuple("SIA", ["tick", "nodes", "coverage", "age_days_min", "age_days_max"])
EDUCATION = namedtuple("EDUCATION", ["tick", "nodes"])
NINE_MONTHS = 274   # 9 months in days (30 day months + 9/12 of the 5 addition days in a year)
SIX_YEARS = 6 * 365 # 6 years in days
todo = [
    SEROSURVEY(9, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 9, nodes 0-5, [2-6) years old
    SIA(10, [1, 3, 5], 0.80, NINE_MONTHS, SIX_YEARS), # Tick 10, nodes 1, 3, and 5, 80% coverage, [2-6) years old
    SEROSURVEY(11, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 11, nodes 0-5, [2-6) years old
    EDUCATION(30, [0, 1, 2, 3, 4, 5]),  # Tick 30, nodes 0-5
    ]

model.nodes.add_vector_property("seronegativity", model.params.ticks, dtype=np.uint32)

@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.uint32[:]), parallel=True)
def _do_serosurvey(count, targets, nodeids, susceptibilities, seronegativity):
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
    if not type(iv) in iv_map:
        raise ValueError(f"Missing invoke function for intervention type: {type(iv).__name__}")


# # **Running the Simulation**
# 
# In[25]:
    do_interventions, # type: ignore
# ## Iterating Over the Simulation Duration
# In[26]:
from tqdm import tqdm
# # Post-Simulation Analysis

# In[27]:
# In[28]:
# In[29]:
# In[30]:
# In[31]:
# In[32]:
# In[33]:
# In[34]:
# In[35]:

# ## Serosurvey Results (Checking the SIA)

# In[36]:


# convert nodes 0-5 and ticks 5-15 of model.nodes.seronegativity to a DataFrame and print its rows

seronegativity = model.nodes.seronegativity[0:6, 0:16]
seronegativity_df = pd.DataFrame(seronegativity.transpose(), columns=[f"Node {i}" for i in range(6)])
print(seronegativity_df[5:16])
print()
print(np.array(seronegativity_df.iloc[11]) / np.array(seronegativity_df.iloc[9]))


# In[37]:


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

# In[38]:


probe_node = 6

istart = initial_populations.sum()
print(f"{istart=:,}")
iend = model.population.count
print(f"{iend=:,}")
node_births = (model.population.nodeid[istart:iend] == probe_node)
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


# In[39]:


import matplotlib.pyplot as plt

# Assuming 'model.population.accessibility' is a NumPy array or a Pandas Series
accessibility_data = model.population.accessibility[istart:iend][node_births]

# Plot the histogram
counts, bins, patches = plt.hist(accessibility_data, bins=3, edgecolor='black')

# Add labels to each bin
for count, bin_edge in zip(counts, bins):
    plt.text(bin_edge, count, str(int(count)), ha='left', va='bottom')

# Plot the histogram
plt.title('Histogram of Accessibility')
plt.xlabel('Accessibility')
plt.ylabel('Frequency')
plt.show()
