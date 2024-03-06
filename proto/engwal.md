# England+Wales Measles Model with Grouped Communities

Uses a Population object to instantiate multiple grouped communities (i.e., all susceptibles are in contiguous memory as are exposed, infectious, and recovered respectively).

Each community is homogeneously well mixed with a gravity model based connectivity between communities.

The model, using the `-n` or `--nodes` parameter, runs the n-largest communities, by population, from the England and Wales data.

## Network

The network is based on the distances between communities (cached on disk) and optional $`k`$, $`a`$, $`b`$, and $`c`$ parameters for the function

```math
k \left( p_1^a p_2^b \over N d^c \right)
```

where $`N`$ is the total population (per @kfrey-idm). The $`k`$, $`a`$, $`b`$, and $`c`$ parameters may be specified on the command line.

## Community Initialization

Population initialization involves creating a `Population` object with the number of nodes, the expected groups of agents, and the required properties on agents (name, type, default).

Initializing the population with `realize()` also requires a callback function to provide instance data for each community, primarily initial populations and optionally, per community data such as demographics.

Communities are initialized with total active (`susceptible` + `recovered`) population equal to the starting population from the England+Wales demographic data. Communities also include an initially inactive `unactive` group to accomodate births and immigration. The `susceptible` population is set to total / R<sub>0</sub>. The remaining population is placed in `recovered`.

In addition, any community with cases detected in the first year according to the historical data, are seeded with one initial infection.

## Running the Model

Running the model is straightforward iterating over the ticks, optionally specified with the `-t` or `--ticks` command line argument, and stepping applying the various "kernels" to update the state of the model:
* vital dynamics
* updating infectious agents
* updating exposed agents
* transmitting infections from infectious agents to susceptible agents
* recording relevant statistics

### Vital Dynamics

Vital dynamics per community are computed daily, so we determine the number of births or immigrations _today_ according the ammortization of births and population increase for the given year. Deaths occur if the population size increases by _less_ than the number of births for the year. Immigration occurs if the population size increases by _more_ than the number of births for the year.

Immigration is weakly modeled as a new recovered individual with an age uniformly drawn from 0-89 years.

**Vital dynamics is calculated based on the `population` and `births` array properties of the communities. If you want to run for more than 20 years you will need to extend these arrays either from additional historical data or with synthetic data.** Otherwise vital dynamics will effectively end after 20 years (no additional births or immigration).

### Infectious Update

Simply decrements the infectious timer and, if it hits zero, moves the agent from the `infectious` group to the `recovered` group.

### Exposed Update

Simply decrements the incubation timer and, if it hits zero, draws for a infectious duration and moves the agent from the `exposed` group to the `infectious` group.

### Transmission

Transmission involves three phases:
1. Get the total contagion for each community (currently == number of infectious agents)
2. Move contagion to/from each community based on the network
3. Expose and potentially infect Poisson(expected infections) agents from the `susceptible` group of the community. Currently we do a uniform random draw to compare against the agent's susceptibility but susceptibility of any susceptible agent is 1 so this is unnecessary. However it provides a placeholder for considering any mitigating factors, e.g. maternal immunity. Infected agents get a draw for their incubation duration and are moved from the `susceptible` group to the `exposed` group.

### Reporting

Currently tracks the size of each group for each community at each tick. Change the `report` structure on the population object (in initialization) in order to allocate space for other reporting and modify the `update_report()` function to record different data.
