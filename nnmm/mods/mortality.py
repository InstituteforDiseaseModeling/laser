import numpy as np
import numba as nb
from datetime import datetime
from tqdm import tqdm
import pdb

from idmlaser.kmcurve import pdsod


# ## Non-Disease Mortality 
# ### Part II
# 
# We import `pdsod` ("predicted_days_of_death") which currently uses a USA 2003 survival table. `pdsod` will draw for a year of death (in or after the current age year) and then will draw a day in the selected year.
# 

def init( model ):
    capacity = model.population.capacity
    population = model.population # to keep code similar
    age = population.age
    dods = population.dod
    tstart = datetime.now(tz=None)  # noqa: DTZ005
    dods[0:population.count] = pdsod(age[0:population.count], max_year=100)
    #dods[0:population.count] = pdsod(age[0:population.count].astype(np.int32)*365, max_year=100)
    tfinish = datetime.now(tz=None)  # noqa: DTZ005
    print(f"Elapsed time for drawing dates of death: {tfinish - tstart}")

    dods -= age.astype(dods.dtype) # renormalize to be relative to _now_ (t = 0)
    print(f"First 32 Ages (should all be positive - these agents were born before today):\n{age[:32]}")
    print(f"First 32 DoDs (should all be positive - these agents will all pass in the future):\n{dods[:32]}")

    if "eula_age" in model.params.__dict__:
        # Sort by age
        print( "Sorting all agents by age." )
        model.population.sort_by_property("age")
        # Convert eula threshold in years to (negative) day-of-birth
        eula_age = model.params.eula_age * 365
        # Find index of first value after eula_age
        split_index = np.searchsorted(model.population.age, eula_age)
        # Get list of indices we would EULA-ify
        dod_filtered = model.population.dod[0:split_index]
        # Convert these all to "simulation year of death"
        death_years = dod_filtered // 365
        # Now invoke function in Population class to calculate the expected deaths by simulation year
        model.population.expected_deaths_over_sim( death_years, split_index=split_index )
        #print( model.population.expected_new_deaths_per_year )
        # Now 'squash' (split) the population to keep only the non-EULA group
        model.population.eliminate_eulas( split_index=split_index )


# ## Non-Disease Mortality 
# ### Part III
# 
# If we had no births, we could use a regular FIFO queue with the dates of death sorted to process non-disease deaths in time order. However, we may have agents born during the simulation who are predicted to die from non-disease causes before the end of the simulation and need to be insert, in sorted order, into the queue. So, we will use a priority queue - a data structure which allows for efficient insertion and removal while maintaining sorted order - to keep track of agents by date of non-disease death.

# In[11]:


    def queue_deaths():
        from idmlaser.utils import PriorityQueuePy

        tstart = datetime.now(tz=None)  # noqa: DTZ005
        dods = population.dod
        mortality = PriorityQueuePy(capacity, dods)
        for i in tqdm(indices := np.nonzero(dods[0:population.count] < model.params.ticks)[0]):
            mortality.push(i)
        tfinish = datetime.now(tz=None)  # noqa: DTZ005

        print(f"Elapsed time for pushing dates of death: {tfinish - tstart}")
        print(f"Non-disease mortality: tracked {len(indices):,}, untracked {population.count - len(indices):,}")

        model.nddq = mortality
    queue_deaths()


# ## Non-Disease Mortality 
# ### Part IV
# 
# During the simulation, we will process all agents with a predicated date of death today marking them as deceased and removing them from the active population count.
# 
# **_TODO:_** Set a state to indicate "deceased" _or_ ignore the priority queue altogether and compare `dod` against `tick` and ignore agents where `dod < tick`.

# In[12]:



def do_non_disease_deaths(model, tick):
    # Add eula population
    year = tick // 365
    if model.population.expected_new_deaths_per_year is not None:
        for nodeid in range(len(model.population.expected_new_deaths_per_year)):
            model.nodes.population[nodeid,tick+1] -= (model.population.expected_new_deaths_per_year[nodeid][year]/365)

    pq = model.nddq
    while len(pq) > 0 and pq.peekv() <= tick:
        i = pq.popi()
        nodeid = model.population.nodeid[i]
        model.nodes.population[nodeid,tick+1] -= 1
        model.nodes.deaths[nodeid,year] += 1

    return


