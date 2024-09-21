from pathlib import Path
import numpy as np
from tqdm import tqdm
import idmlaser_cholera.pyramid as pyramid
from .. import manifest

# ## Non-Disease Mortality 
# ### Part I
# 
# We start by loading a population pyramid in order to initialize the ages of the initial population realistically.
# 
# The population pyramid is typically in 5 year age buckets. Once we draw for the age bucket, we draw uniformly for a date of birth within the range of the bucket.
# 
# **Note:** the values in `model.population.dob` are _positive_ at this point. Later we will negate them to convert them to dates of birth prior to now (t = 0).
def init( model ):
    initial_populations = model.nodes.population[:,0]
    capacity = model.population.capacity
    with manifest.age_data as pyramid_file:
        print(f"Loading pyramid from '{pyramid_file}'...")
        # Convert it to a string if needed
        age_distribution = pyramid.load_pyramid_csv(pyramid_file)

        print("Creating aliased distribution...")
        aliased_distribution = pyramid.AliasedDistribution(age_distribution[:,4])
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
        model.population.dob[mask] = np.random.randint(low=minimum_age[i], high=limit_age[i], size=mask.sum())

