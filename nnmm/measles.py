import sys
import pdb
from pathlib import Path
import numpy as np
    "ticks": int(365*10),
    "eula_age": 5
    "ri_coverage": np.float32(0.75),
model.params = PropertySet(meta_params, measles_params, network_params) # type: ignore
# In[4]:
# NOT REALLY NEEDED IF USING CACHED INIT POP
# Add a property for node id. 419 nodes requires 9 bits so we will allocate a 16 bit value. Negative IDs don't make sense, so, `uint16`.
# In[5]:
# NOT REALLY NEEDED IF USING CACHED INIT POP
# Default data type is uint32.
# In[6]:
nodes = Population(capacity=node_count) # TODO: rename `Population` to some appropriate to agents _and_ nodes
# In[7]:
# In[8]:
# Adding ri_timer here since it's referred to in do_births.
model.population.add_scalar_property("ri_timer", np.uint16)

        model.nodes.births[:, year] = np.random.poisson(model.nodes.population[:, tick] * model.params.cbr / 1000)
        #print( f"Births for year {year} = {model.nodes.births[:, year]}" )
    model.population.dod[istart:iend] = pdsod(model.population.dob[istart:iend], max_year=100)   # make use of the fact that dob[istart:iend] is currently 0
    # Randomly set ri_timer for coverage fraction of agents to a value between 8.5*30.5 and 9.5*30.5 days
    # change these numbers or parameterize as needed
    ri_timer_values = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16)

    # Create a mask to select coverage fraction of agents
    # Do coverage by node, not same for every node
    try:
        mask = np.random.rand(count_births) < (model.nodes.ri_coverages[model.population.nodeid[istart:iend]])
    except Exception as ex:
        pdb.set_trace()

    # Set ri_timer values for the selected agents
    model.population.ri_timer[istart:iend][mask] = ri_timer_values[mask]

    set_accessibility(model, istart, iend )
    add_with_ips( model, count_births, istart, iend )
    add_maternal_immunity( model, istart, iend )
# ## Non-Disease Mortality 
# ### Part I
# In[9]:

# When do we first have data to check if we have a cached input file that matches our parameters?
from idmlaser.kmcurve import cumulative_deaths
def check_for_cached():
    from idmlaser.numpynumba.population import check_hdf5_attributes
    cached = check_hdf5_attributes(
            hdf5_filename="pop_init_eulagized.h5",
            initial_populations=initial_populations,
            age_distribution=age_distribution,
            cumulative_deaths=cumulative_deaths,
            eula_age=model.params.eula_age
        )
# Just invoked if user comments this line in. Working on automation.
#check_for_cached()

aliased_distribution = pyramid.AliasedDistribution(age_distribution[:,4])
print(f"Sampling {count_active:,} ages... {population.count=:,}")
buckets = aliased_distribution.sample(population.count)
population.add_scalar_property("dob", np.int32)
    population.dob[mask] = np.random.randint(low=minimum_age[i], high=limit_age[i], size=mask.sum())
# ## Non-Disease Mortality 
# ### Part II
# In[10]:
dods[0:population.count] = pdsod(dobs[0:population.count], max_year=100)
# Now we are going to eula-ify our population at age=5.0
if "eula_age" in model.params.__dict__:
    model.population.init_eula(model.params.eula_age)

# ## Non-Disease Mortality 
# ### Part III
# In[11]:
# ## Non-Disease Mortality 
# ### Part IV
# In[12]:
    # Add eula population
    if model.population.expected_new_deaths_per_year is not None:
        for nodeid in range(len(model.population.expected_new_deaths_per_year)):
            model.nodes.population[nodeid,tick+1] -= (model.population.expected_new_deaths_per_year[nodeid][year]/365)
        nodeid = model.population.nodeid[i]
        model.nodes.population[nodeid,tick+1] -= 1
        model.nodes.deaths[nodeid,year] += 1
# In[13]:
# In[14]:
# ## Routine Immunization (RI) 
# Do MCV1 for most kids at around 9mo. Do MCV2 for most of the kids who didn't take around 15mo. There are a bunch of ways of doing this.
# What we're doing here is using an IP -- to borrow EMOD terminology -- for Accessibility, with values 0 (Easy), 1 (Medium), 2 (Hard). See elsewhere for that.
# The MCV1 Timer is a uniform draw from 8.5 months to 9.5 months. MCV2 Timer is a uniform draw from 14.5 months to 15.5 months.
# ### Coverages
# In[15]:
nodes.add_scalar_property("ri_coverages", dtype=np.float32)
nodes.ri_coverages = np.random.rand(node_count)
# ## RI
# Vary as needed.
# In[16]:
    # Get the coverage probabilities for the relevant agents
    coverages = model.nodes.ri_coverages[model.population.nodeid[istart:iend]]                        
    
    random_values = np.random.rand(coverages.size)
    
    # Calculate attenuated probabilities
    prob_0 = 0.85 * coverages                                           
    prob_1 = 0.14 * coverages                                   
    prob_2 = 0.01 * coverages
    
    # Normalize probabilities to ensure they sum to 1 after attenuation
    total_prob = prob_0 + prob_1 + prob_2
    prob_0 /= total_prob
    prob_1 /= total_prob
    prob_2 /= total_prob
            
    # Initialize accessibility array with zeros      
    accessibility = np.zeros_like(random_values, dtype=np.uint8)
    
    accessibility[random_values < prob_0] = 0                                   
    accessibility[(random_values >= prob_0) & (random_values < prob_0 + prob_1)] = 1
    accessibility[random_values >= prob_0 + prob_1] = 2
    
    # Assign to the population's accessibility attribute
    model.population.accessibility[istart:iend] = accessibility
# ## RI 
# ### Add based on accessibility group for newborns
# In[17]:
def add_with_ips(model, count_births, istart, iend):
    ri_timer_values_9mo = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16) # 9mo-ish
    ri_timer_values_15mo = np.random.uniform(14.5 * 30.5, 15.5 * 30.5, count_births).astype(np.uint16) # 15mo-ish                                                                                           

    # Get the accessibility values for the current range
    accessibility = model.population.accessibility[istart:iend]                                       
                                                                                                      
    # Masks based on accessibility                                                                    
    mask_0 = (accessibility == 0)
    mask_1 = (accessibility == 1)
    mask_2 = (accessibility == 2) # for validation                                                    
    if np.count_nonzero( mask_1 ) == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 1 (medium)." )                
    if np.count_nonzero( mask_2 ) == 0:
            
    # mask_2 is unnecessary since we don't apply any timer for it                                     
        
    # Apply the 9-month-ish timer to accessibility 0
    model.population.ri_timer[istart:iend][mask_0] = ri_timer_values_9mo[mask_0]                      
    
    # Apply the 15-month-ish timer to accessibility 1                                                 
    model.population.ri_timer[istart:iend][mask_1] = ri_timer_values_15mo[mask_1]                     
    
    # No need to apply anything to accessibility 2, as it should remain unmodified                    
    
    return                                                                                            


# ## RI
# ### "Step-Function"
# Timers get counted down each timestep and when they reach 0, susceptibility is set to 0.
# In[18]:
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.int32[:], nb.int64), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility, dob, tick):
        if ri_timer[i] > 0:
            ri_timer[i] -= 1
            if ri_timer[i] == 0:
    # Ensure arrays are of types supported by Numba
    #count = np.int64(model.population.count)
    #ri_timer = model.population.ri_timer.astype(np.uint16)
    #susceptibility = model.population.susceptibility.astype(np.uint8)
    #dob = model.population.dob.astype(np.int32)
    #tick = np.int64(tick)
    #pdb.set_trace()
    _update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility, model.population.dob, tick)
# # Maternal Immunity (Waning)
# All newborns come into the world with susceptibility=0. They call get a 6month timer. When that timer hits 0, they become susceptible.
# In[19]:
model.population.add_scalar_property("susceptibility_timer", np.uint8)
    # enable this after adding susceptibility property to the population (see cells below)            
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity                
    model.population.susceptibility_timer[istart:iend] = int(0.5*365) # 6 months                      
# Update
# Define the function to decrement susceptibility_timer and update susceptibility
# ## Initial Infections
# In[20]:
# initial_infections = np.random.randint(0, 11, model.nodes.count, dtype=np.uint32)
initial_infections = np.uint32(np.round(np.random.poisson(prevalence*initial_populations)))
# Print this _before_ initializing infections because `initial_infections` is modified in-place.
# We initialize $n_{ij} = n_{ji} = k \frac {P_i^a \cdot P_j^b} {D_{ij}^c}$
# In[21]:

network /= np.power(initial_populations.sum(), c)    # normalize by total population^2
# In[22]:
# ## SIAs
# In[23]:
sias = [(10, [1, 3, 5], 0.80)]  # Tick 10, nodes 1, 3, and 5, 80% coverage.
@nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32), parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, coverage):
                if np.random.random_sample() < coverage:
                    susceptibilities[i] = 0
def do_sias(model, tick):
    while len(sias) > 0 and sias[0][0] == tick:
        campaign = sias.pop(0)
        print(f"Running SIA {campaign=} at tick {tick}")
        (day, nodes, coverage) = campaign
        targets = np.zeros(model.nodes.count, dtype=np.uint8)
        targets[nodes] = 1
        _do_sia(model.population.count, targets, model.population.nodeid, model.population.susceptibility, coverage)
# In[24]:
    do_sias, # type: ignore
# ## Running the Simulation
# In[25]:

import numpy as np
def save_initial_pop():
    # Need to save some metadata/data that was critical to the creation of the init population
    # initial population is at: model.nodes.population[:,0]
    # age_distribution
    # Generate a unique filename with the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"laser_cache/pop_init_eulagized_{timestamp}.h5"
    model.population.save( filename, initial_populations=initial_populations, age_distribution=age_distribution, cumulative_deaths=cumulative_deaths, eula_age=model.params.eula_age )
# Just invoked if user comments this line in. Working on automation.
#save_initial_pop()
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]: