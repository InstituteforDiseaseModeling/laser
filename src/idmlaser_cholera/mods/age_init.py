from pathlib import Path
import numpy as np
from tqdm import tqdm
import laser_core.demographics.pyramid as pyramid
import pdb

# age_init.py
import json  # Assuming the age data is stored in a JSON file; adjust as necessary.

# This get way more involved than I wanted when I needed to load the age_distribution from the user-specifiec path early enough to check whether the 
# cached models match the distribution before creating a new model.

class AgeDataManager:
    def __init__(self):
        self._path = None
        self._data = None

    def get_data(self):
        """Load and return the age data."""
        if self._data is None:
            if self._path is None:
                raise ValueError("Path not set. Please use set_path to specify a file path.")
            self._data = pyramid.load_pyramid_csv(Path(self._path))
        return self._data

    def set_path(self, path):
        """Set the file path for the age data."""
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        self._path = path
        self._data = None  # Clear any previously loaded data

# Singleton instance for module-level access
age_data_manager = AgeDataManager()


# ## Non-Disease Mortality 
# ### Part I
# 
# We start by loading a population pyramid in order to initialize the ages of the initial population realistically.
# 
# The population pyramid is typically in 5 year age buckets. Once we draw for the age bucket, we draw uniformly for a date of birth within the range of the bucket.
# 
# **Note:** the values in `model.population.dob` are _positive_ at this point. Later we will negate them to convert them to dates of birth prior to now (t = 0).
def init( model, manifest ):

    print(f"Loading pyramid from '{manifest.age_data}'...")
    # Convert it to a string if needed
    age_distribution = age_data_manager.get_data()

    #initial_populations = model.nodes.population[:,0]
    if model.nodes is None:
        raise RuntimeError( "nodes does not seem to be initialized in model object." )
    if model.nodes.population is None:
        raise RuntimeError( "nodes.population does not seem to be initialized in model object." )
    initial_populations = model.nodes.population[0]
    capacity = model.population.capacity

    print("Creating aliased distribution...")
    #aliased_distribution = pyramid.AliasedDistribution(age_distribution[:,4])
    aliased_distribution = pyramid.AliasedDistribution(age_distribution[4])
    count_active = initial_populations.sum()

    print(f"Sampling {count_active:,} ages... {model.population.count=:,}")
    buckets = aliased_distribution.sample(model.population.count)
    minimum_age = age_distribution[:, 0] * 365      # closed, include this value
    limit_age = (age_distribution[:, 1] + 1) * 365  # open, exclude this value
    mask = np.zeros(capacity, dtype=bool)

    print("Converting age buckets to ages...")
    for i in tqdm(range(len(age_distribution))):
        mask[:count_active] = (buckets == i)    # indices of agents in this age group bucket
        # draw uniformly between the start and end of the age group bucket
        try:
            model.population.dob[mask] = np.random.randint(low=minimum_age[i], high=limit_age[i], size=mask.sum())
        except Exception as ex:
            print( str( ex ) )
            print( f" at i={i}." )
            pdb.set_trace()
        # Not negative actually
        model.population.age[mask] = model.population.dob[mask]

