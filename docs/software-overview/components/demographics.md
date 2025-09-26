# Demographics

<!--Need to an an intro, explaining generalizations about demographics for LASER.

ALL topics about demographics will go here:

- Age structure
- Births
- Deaths
- All the vital dynamics sections??
- population pyramids
- kapalan-meier estimator
- etc

Just pasted in content from the various topics; this will need an intro and text explaining how to configure these, how they fit together (eg, when to use age pyrmaids, when to use kaplan-meier); also code blocks, etc need to be formatted

SO: Vital dynamics model: that section probably needs its own topic page, since it's long. Still kind of confused about what a "model" is in terms of laser components--is this a stand-alone model? Or the 'piece' that implements vital dynamics within an actual model? The answer to that will determine where we put the VDM topic page in the TOC.

Also, make sure to link to appropriate topics! (esp parameters) -->

## Age structure

If you want to work with age structure for a short simulation which doesn’t need births you can just give everyone an age (based on distribution) and increment it each timestep. The `laser_core.demographics.pyramid` module is provided to support the initialization of agents with plausible initial ages.

### Births

#### Preborn management in LASER

LASER’s design philosophy emphasizes contiguous and fixed-size arrays, meaning all agents—both currently active and preborn—are created at the start of the simulation. Preborns are “activated” as they are born, rather than being dynamically added. Several approaches to handling preborns while adhering to these principles are outlined below:

Negative and Positive Birthdays:

- Assign `date_of_birth` values in the past (negative) for active agents.
- Assign `date_of_birth` values in the future (positive) for preborns.

Unified Preborn Marker:

- Set all preborns’ `date_of_birth` to a placeholder value (e.g., -1).
- Update the `date_of_birth` to the current timestep when a preborn is born.

Active Flag Only (if not modeling age structure):

- If the model doesn’t require age structure, you can skip date_of_birth entirely. Instead, use an active flag. Preborns start with `active = False` and are switched to `active = True` during the fertility step. This simplifies implementation while remaining consistent with LASER principles.

#### Calculating age from birthday

If calculating age isn’t frequent or essential, you can avoid explicitly tracking an age property. Instead, compute age dynamically as the difference between the current timestep (now) and `date_of_birth`. For models that depend on age-specific dynamics (e.g., fertility rates by age group), consider adding a dedicated age property that updates at each timestep.

### Deaths

The recommended way of doing mortality in LASER is by precalculating a lifespan for each agent, rather than probabilistically killing agents as the simulation runs. This can take different forms: If you prefer to track agent age, you can also have an agent lifespan. Alternatively, if you are just using `date_of_birth` you can have a `date_of_death`, where theses ‘dates’ are really simulation times (‘sim day of birth’ and ‘sim day of death’).

In LASER, we strive to leave the contiguous arrays of agent data in place, without adding or deleting elements (allocating or freeing). This means that to model mortality, we prefer to ‘kill’ agents by doing either:

 1. check that their age is greater than their lifespan (or that the current timestep is greater than their ‘sim day of death’) in each component that cares, or

 2. Set an active flag to "false" or a dead flag to "true."

 The second approach is simpler, and avoids doing millions of comparison operations, at the cost of an additional property. Note that many component operations (step functions) can be done without checking whether the agent is alive, because, for example, as long as transmission never infects a dead person, decrementing all non-zero infection timers will only operate on live agents.

 Finally, while you can set lifespans using any algorithm you want, `laser_core.demographics.kmestimator` is provided to support these calculations.


## Population pyramids

The `AliasedDistribution` class provides a way to sample from a set of options with unequal probabilities, e.g., a population pyramid.

The input to the `AliasedDistribution` constructor is an array of counts by bin as we would naturally get from a population pyramid (# of people in each age bin).

`AliasedDistribution.sample()` returns bin indices so it is up to the user to convert the values returned from `sample()` to actual ages.

Expected format of the population pyramid CSV file for `load_pyramid_csv()`:

```
Header: Age,M,F
start-end,#males,#females
start-end,#males,#females
start-end,#males,#females
…
start-end,#males,#females
max+,#males,#females
```

For example,

```
Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,10274310
15-19,11448281,10950664
…
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
```

`load_pyramid_csv()` returns a 4 column NumPy array with the following columns:

```
0 - Lower bound of age bin, inclusive
1 - Upper bound of age bin, inclusive
2 - number of males in the age bin
3 - number of females in the age bin
```

/// details | Code example: Loading population pyramids from .csv files

```
import numpy as np
from laser_core.demographics import load_pyramid_csv, AliasedDistribution
import importlib.util
import os

MCOL = 2
FCOL = 3

MINCOL = 0
MAXCOL = 1

# Access the bundled file dynamically
laser_core_path = importlib.util.find_spec("laser_core").origin
laser_core_dir = os.path.dirname(laser_core_path)
pyramid_file = os.path.join(laser_core_dir, "data/us-pyramid-2023.csv")

pyramid = load_pyramid_csv(pyramid_file)
sampler = AliasedDistribution(pyramid[:, MCOL])    # We'll use the male population in this example.
n_agents = 100_000
samples = sampler.sample(n_agents)              # Sample 100,000 people from the distribution.
# samples will be bin indices, so we need to convert them to ages.
bin_min_age_days = pyramid[:, MINCOL] * 365          # minimum age for bin, in days (include this value)
bin_max_age_days = (pyramid[:, MAXCOL] + 1) * 365    # maximum age for bin, in days (exclude this value)
mask = np.zeros(n_agents, dtype=bool)
ages = np.zeros(n_agents, dtype=np.int32)
for i in range(len(pyramid)):   # for each possible bin value...
    mask[:] = samples == i      # ...find the agents that belong to this bin
    # ...and assign a random age, in days, within the bin
    ages[mask] = np.random.randint(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

# in some LASER models we convert current ages to dates of birth by negating the age
# dob = -ages
```
///

To explore working with age pyramids, see the [Age Pyramid Examples](../../tutorials/age_pyramid.ipynb) in the [Tutorials](../../tutorials/index.md) section.


## Kaplan-Meier estimators

The `KaplanMeierEstimator` is used to predict age or year of death. It takes an array of cumulative deaths and returns an object that will sample from the Kaplan-Meier distribution.

A sample input array of cumulative deaths might look like this:

```
cd[0] = 687 # 687 deaths in the first year (age 0)
cd[1] = 733 # +46 deaths in the second year (age 1)
cd[2] = 767 # +34 deaths in the third year (age 2)
...
cd[100] = 100_000  # 100,000 deaths by end of year
```

`predict_year_of_death()` takes an array of current ages (in years) and returns an array of predicted years of death based on the cumulative deaths input array.

!!! note
    `predict_year_of_death()` can use non-constant width age bins and will return predictions by age bin. In this case, it is up to the user to convert the returned bin indices to actual years.

A sample non-constant width age bin input array might look like this:

```
cd[0] = 340 # 1/2 of first year deaths in the first 3 months
cd[1] = 510 # another 1/4 (+170) of first year deaths in the next 3 months
cd[2] = 687 # another 1/4 (+177) of first year deaths in the last 6 months
cd[3] = 733 # 46 deaths in the second year (age 1)
cd[4] = 767 # 34 deaths in the third year (age 2)
...
cd[103] = 100_000  # 100,000 deaths by end of year 100
```

In this example, values returned from predict_year_of_death() would need to be mapped as follows:

```
0 -> (0, 3] months
1 -> (3, 6] months
2 -> (6, 12] months
3 -> 1 year
4 -> 2 years
...
102 -> 100 years
```

`predict_age_at_death()` takes an array of current ages (in days) and returns an array of predicted ages (in days) at death. The implementation assumes that the cumulative deaths input array to the estimator represents one year age bins. If you are using non-constant width age bins, you should manually convert bin indices returned from `predict_year_of_death()` to ages.


## Spatial distributions of populations

<!-- [Added here since it's part of the demographics subpackage, but it might make more sense with the migration information] -->