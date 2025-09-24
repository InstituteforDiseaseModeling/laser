# Migration

<!-- ADD INTRO. Explain how LASER has the various types of migration models, that the user can select how to implement migration using the following types. (Will need to include the name of the component, and a link to the section in the API docs that defines it). Make sure each type of migration includes info on how laser implements it/ties into the code. Currently just contains content ported from sphinx docs. -->

The ability to add spatial dynamics to LASER models is one of the features that makes the framework so powerful. There are multiple methods available for adding migration to your model, and the class you choose will depend on which features are important for your research question. Each of the migration models will distribute your population of agents among a set of nodes, with set distances between nodes, and utilize a matrix to define the connection between nodes. How agents or infectivity travels between the nodes (and which nodes they may travel to) will be determined by the specific migration model you choose.

<!-- we can add the model selection considerations here if they're relevant to the docs...currently undecided if we want them here, or if we should create a "migration" section in the ModelHub and just link to that -->

## Sequential migration matrix

<!-- This is used in the SIR example, so we need to include a section on it here. -->
[Agents move sequentially from node to node in a chain]

## Gravity model

The [gravity model](https://en.wikipedia.org/wiki/Gravity_model_of_migration) can be used to compute the migration of people between nodes located at specific distances, with migration rates proportional to population size and the distance between nodes. This type of migration is useful when you would like to add 2-dimensional movement of agents to nodes.

Functional form:

$$
M_{ij} = k \frac{P_i^{a} P_j^{b}}{d_{ij}^{c}}
$$

Where:

- $M_{ij}$ = migration flow from origin i to destination j
- $P_i, P_j$ = populations of origin and destination
- $d_{ij}$ = distance between i and j
- $k$ = Parameter: scaling constant
- $a$ = Parameter: exponent for the population of the origin
- $b$ = Parameter: exponent for the population of the destination
- $c$ = Parameter: exponent for the distance



The following example demonstrates implementing the gravity model to calculate the number of migrants moving between nodes. Agents are randomly assigned to different migration paths.

```
import numpy as np
from laser_core.migration import gravity

# Define populations and distances
populations = np.array([5000, 10000, 15000, 20000, 25000])  # Unequal populations
distances = np.array([
    [0.0, 10.0, 15.0, 20.0, 25.0],
    [10.0, 0.0, 10.0, 15.0, 20.0],
    [15.0, 10.0, 0.0, 10.0, 15.0],
    [20.0, 15.0, 10.0, 0.0, 10.0],
    [25.0, 20.0, 15.0, 10.0, 0.0]
])

# Gravity model parameters
k = 0.1    # Scaling constant
a = 0.5    # Exponent for the population of the origin
b = 1.0    # Exponent for the population of the destination
c = 2.0    # Exponent for the distance

# Compute the gravity model network
migration_network = gravity(populations, distances, k=k, a=a, b=b, c=c)

# Normalize to ensure total migrations represent 1% of the population
total_population = np.sum(populations)
migration_fraction = 0.01  # 1% of the population migrates
scaling_factor = (total_population * migration_fraction) / np.sum(migration_network)
migration_network *= scaling_factor

# Generate a node ID array for agents
node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(populations)])

# Initialize a 2D array for migration counts
migration_matrix = np.zeros_like(distances, dtype=int)

# Select migrants based on the gravity model
for origin in range(len(populations)):
    for destination in range(len(populations)):
        if origin != destination:
            # Number of migrants to move from origin to destination
            num_migrants = int(migration_network[origin, destination])
            # Select migrants randomly
            origin_ids = np.where(node_ids == origin)[0]
            selected_migrants = np.random.choice(origin_ids, size=num_migrants, replace=False)
            # Update the migration matrix
            migration_matrix[origin, destination] = num_migrants
```


## The competing destinations model

The competing destinations model extends the gravity model by incorporating the fact that nodes may not be independent, but instead may "compete" with each other for incoming agents. Some nodes may be more or less attractive than other nodes, regardless of their proximity to the origin node. There may  be synergistic or antagonistic effects of nodes, creating specific networks and relationships among a series of nodes.

For example, in a “synergistic” version, perhaps migratory flow from Boston to Baltimore is higher than flow between two comparator cities of similar population and at similar distance, because the proximity of Washington, D.C. to Baltimore makes travel to Baltimore more attractive to Bostonians. This would be accounted for by a positive value of $\delta$. On the other hand, this term may also be “antagonistic” if Washington is such an attractive destination that Bostonians eschew travel to Baltimore entirely; this would indicate a negative value of $\delta$.

Functional form:

$$
M_{i,j} = k \frac{P_i^{a} P_j^{b}}{d_{ij}^{c}} \left(\sum_{k \neq i,j} \frac{P_k^{b}}{d_{jk}^{c}}\right)^{\delta}
$$



## Stouffer's rank model

[Stouffer](https://doi.org/10.2307/2084520) argued that human mobility patterns do not respond to absolute distance directly, but only indirectly through the accumulation of intervening opportunities for destinations. Stouffer thus proposed a model with no distance-dependence at all, rather only a term that accounts for all potential destinations closer than destination $j$; thus, longer-distance travel depends on the density of attractive destinations at shorter distances.

Mathematical formulation:

Define $\Omega (i,j)$ to be the set of all locations $k$ such that $D_{i,k} \leq D_{i,j}$

$$
M_{i,j} = kp_i^a \sum_j \left(\frac{p_j}{\sum_{k \in D(i,j)} p_k}\right)^b
$$

This presents us with the choice of whether or not the origin population $i$ is included in $\Omega$ - i.e., does the same "gravity" that brings others to visit a community reduce the propensity of that community's members to travel to other communities?

The Stouffer model **does not** include the impact from the local community:

$$
\Omega(i,j) = \left(k:0 < D_{i,k} \leq D_{i,j}\right).
$$

The Stouffer variant model **does** include the impact of the local community:

$$
\Omega(i,j) = \left(k:0 \leq D_{i,k} \leq D_{i,j}\right).
$$

To simplify the code, `laser-core`'s implementation of the Stouffer model includes a parameter `include_home`.



## Radiation model

The [radiation model](https://www.nature.com/articles/nature10856) is a parameter-free model (up to an overall scaling constant for total migration flux), derived from arguments around job-related commuting. The radiation model overcomes limitations of the gravity model (which is limited to flow at two specific points and is proportional to the populations at source and destination) by only requiring data on population densities. It can describe situations in which outbound migration flux from origin to destination is enhanced by destination population and absorbed by the density of nearer destinations.

Mathematical formulation, whith $\Omega$ defined as above in the Stouffer model:

$$
M_{i,j} = k \frac{p_i p_j}{\left(p_i + \sum_{k \in \theta(i,j)} p_k\right) \left(p_i + p_j + \sum_{k \in \theta(i,j)} p_k\right)}
$$

We again use the parameter `include_home` to determine whether or not location $i$ is to be included in $\Omega(i,j)$.
