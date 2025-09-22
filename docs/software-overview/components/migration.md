# Migration

ADD INTRO. Explain how LASER has the variout types of migration models, that the user can select how to implement migration using the following types.


## Gravity model

<!-- is this the "gravity migration model matrix" referred to in the "simple spatial SIR model with synthetic data" section? If so, is it an importable component?? -->

The gravity model [link to a good source on GM] can be used to compute the migration of people between nodes located at specific distances.


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

## Sequential migration matrix

Saw a reference to this in the "simple spatial SIR model with synthetic data" section, so need info on it.

## The competing destinations model

## Stouffer's rank model

## Radiation model