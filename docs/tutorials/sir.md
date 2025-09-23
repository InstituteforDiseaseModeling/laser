# Building SIR Models

One of the simplest and most commonly used models to describe the progression of an outbreak or epidemic is the SIR (Susceptible - Infected - Recovered) model. We can use the SIR model to explore how to use the LASER framework, staring with a basic SIR model and adding complexity.

This tutorial will:

- Demonstrate how the `LASERframe` and `PropertySet` libraries are used
- Structure a basic disease transmission framework
- Track and visualize results

As you progress through the sections, you will learn how to add spatial dynamics and migration into the disease model, using both synthetic and real-world data.

<!-- would be nice to make this a notebook! -->


## Simple SIR

The SIR model presented here simulates disease dynamics within a closed population in a single node using the `LASERFrame` framework. The population starts with a defined number of susceptible and infected individuals, progresses over time with recovery and transmission components, and tracks results for visualization. This example serves as a practical guide for modeling simple epidemic dynamics. This simple example does not include vital dynamics, age-structured populations, vaccination, or other complex interactions.

### Model components

The `SIRModel` class is the core of the implementation. It initializes a population using `LaserFrame`, sets up disease state and recovery timer properties, and tracks results across timesteps.

/// details | Code example: Implementing `SIRModel`

```
import numpy as np
import matplotlib.pyplot as plt
from laser_core import LaserFrame
from laser_core import PropertySet

class SIRModel:
    def __init__(self, params):
        # Model Parameters
        self.params = params

        # Initialize the population LaserFrame
        self.population = LaserFrame(capacity=params.population_size,initial_count=params.population_size)

        # Add disease state property (0 = Susceptible, 1 = Infected, 2 = Recovered)
        self.population.add_scalar_property("disease_state", dtype=np.int32, default=0)

        # Add a recovery timer property (for intrahost progression, optional for timing)
        self.population.add_scalar_property("recovery_timer", dtype=np.int32, default=0)

        # Results tracking
        self.results = LaserFrame( capacity = 1 ) # number of nodes
        self.results.add_vector_property( "S", length=params["timesteps"], dtype=np.float32 )
        self.results.add_vector_property( "I", length=params["timesteps"], dtype=np.float32 )
        self.results.add_vector_property( "R", length=params["timesteps"], dtype=np.float32 )

        # Components
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def track_results(self, tick):
        susceptible = (self.population.disease_state == 0).sum()
        infected = (self.population.disease_state == 1).sum()
        recovered = (self.population.disease_state == 2).sum()
        total = self.population.count
        self.results.S[tick] = susceptible / total
        self.results.I[tick] = infected / total
        self.results.R[tick] = recovered / total

    def run(self):
        for tick in range(self.params.timesteps):
            for component in self.components:
                component.step()
            self.track_results(tick)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.results.S, label="Susceptible (S)", color="blue")
        plt.plot(self.results.I, label="Infected (I)", color="red")
        plt.plot(self.results.R, label="Recovered (R)", color="green")
        plt.title("SIR Model Dynamics with LASER Components")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Fraction of Population")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("gpt_sir.png")
```
///

The `IntrahostProgression` class manages recovery dynamics by updating infected individuals based on a given recovery rate.

/// details | Code example: Implementing `IntrahostProgression`

```
class IntrahostProgression:
    def __init__(self, model):
        self.population = model.population

        # Seed the infection
        num_initial_infected = int(0.01 * params.population_size)  # e.g., 1% initially infected
        infected_indices = np.random.choice(params.population_size, size=num_initial_infected, replace=False)
        self.population.disease_state[infected_indices] = 1

        # Initialize recovery timer for initially infected individuals
        initially_infected = self.population.disease_state == 1
        self.population.recovery_timer[initially_infected] = np.random.randint(5, 15, size=initially_infected.sum())

    def step(self):
        infected = self.population.disease_state == 1

        # Decrement recovery timer
        self.population.recovery_timer[infected] -= 1

        # Recover individuals whose recovery_timer has reached 0
        recoveries = infected & (self.population.recovery_timer <= 0)
        self.population.disease_state[recoveries] = 2
```
///

The `Transmission` class manages disease spread by modeling interactions between susceptible and infected individuals.

/// details | Code example: Implementing `Transmission`

```
class Transmission:
    def __init__(self, model):
        self.population = model.population
        self.infection_rate = model.params.infection_rate

    def step(self):
        susceptible = self.population.disease_state == 0
        infected = self.population.disease_state == 1

        num_susceptible = susceptible.sum()
        num_infected = infected.sum()
        population_size = len(self.population)

        # Fraction of infected and susceptible individuals
        fraction_infected = num_infected / population_size

        # Transmission logic: Probability of infection per susceptible individual
        infection_probability = self.infection_rate * fraction_infected

        # Apply infection probability to all susceptible individuals
        new_infections = np.random.rand(num_susceptible) < infection_probability

        # Set new infections and initialize their recovery_timer
        susceptible_indices = np.where(susceptible)[0]
        newly_infected_indices = susceptible_indices[new_infections]
        self.population.disease_state[newly_infected_indices] = 1
        self.population.recovery_timer[newly_infected_indices] = np.random.randint(5, 15, size=newly_infected_indices.size)  # Random recovery time
```
///

The simulation parameters are defined using the `PropertySet` class.

/// details | Code example: Defining simulation parameters using `PropertySet`

```
params = PropertySet({
    "population_size": 100_000,
    "infection_rate": 0.3,
    "timesteps": 160
})
```
///

The model is initialized with the defined parameters, components are added, and the simulation is run for the specified timesteps. Results are then visualized.

/// details | Code example: Intiailize, run the simulation, and plot the results

```
# Initialize the model
sir_model = SIRModel(params)

# Initialize and add components
sir_model.add_component(IntrahostProgression(sir_model))
sir_model.add_component(Transmission(sir_model))

# Run the simulation
sir_model.run()

# Plot results
sir_model.plot_results()
```
///

## Spatial SIR

Building upon the simple SIR  model created above, we can add spatial complexity to the framework. Here the simple SIR model will spread the population across 20 nodes. The nodes are arranged in a one-dimensional chain and infection spreads spatially from node 0 as agents migrate; migration is based on a migration matrix.

Two [migration options](../software-overview/components/migration.md) are available:

1. Sequential migration matrix: Agents can only move to the next node in the chain.
2. Gravity model migration matrix: Agents can move in a two-dimensional spatial dynamic, where migration probabilities depend on node distances and population sizes.

In this example, the population is distributed across nodes using a rural-urban skew, and migration timers are assigned to control agent migration frequency.

### Model components

As above, the model will require the use of `LaserFrame`, but will now also include spatial components.

/// details | Code example: Initial model importations

```
import numpy as np
import matplotlib.pyplot as plt
import csv
from laser_core.laserframe import LaserFrame
from laser_core.demographics.spatialpops import distribute_population_skewed as dps
from laser_core.migration import gravity
```
///

Instead of using the `SIRModel`, we will use the `MultiNodeSIRModel`.

/// details | Code example: Creating a model using `MultiNodeSIRModel`

```
# Define the model
class MultiNodeSIRModel:
    """
    A multi-node SIR (Susceptible-Infected-Recovered) disease transmission model.

    Attributes:
        params (dict): Configuration parameters for the model.
        nodes (int): Number of nodes in the simulation.
        population (LaserFrame): Represents the population with agent-level properties.
        results (np.ndarray): Stores simulation results for S, I, and R per node.
        components (list): List of components (e.g., Migration, Transmission) added to the model.
    """

    def __init__(self, params):
        """
        Initializes the SIR model with the given parameters.

        Args:
            params (dict): Dictionary containing parameters such as population size,
                           number of nodes, timesteps, and rates for transmission/migration.
        """
        self.components = []
        self.params = params
        self.nodes = params["nodes"]
        self.population = LaserFrame(capacity=params["population_size"], initial_count=params["population_size"])

        # Define properties
        self.population.add_scalar_property("node_id", dtype=np.int32)
        self.population.add_scalar_property("disease_state", dtype=np.int32, default=0)  # 0=S, 1=I, 2=R
        self.population.add_scalar_property("recovery_timer", dtype=np.int32, default=0)
        self.population.add_scalar_property("migration_timer", dtype=np.int32, default=0)

        # Initialize population distribution
        node_pops = dps(params["population_size"], self.nodes, frac_rural=0.3)
        node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(node_pops)])
        np.random.shuffle(node_ids)
        self.population.node_id[:params["population_size"]] = node_ids

        # Reporting structure: Use LaserFrame for reporting
        self.results = LaserFrame( capacity=self.nodes ) # not timesteps for capacity
        for state in ["S", "I", "R"]:
            self.results.add_vector_property(state, length=params["timesteps"], dtype=np.int32)

        # Record results: reporting could actually be a component if we wanted. Or it can be
        # done in a log/report function in the relevant component (e.g., Transmission)
        self.results.S[self.current_timestep, :] = [
            np.sum(self.population.disease_state[self.population.node_id == i] == 0)
            for i in range(self.nodes)
        ]
        self.results.I[self.current_timestep, :] = [
            np.sum(self.population.disease_state[self.population.node_id == i] == 1)
            for i in range(self.nodes)
        ]
        self.results.R[self.current_timestep, :] = [
            np.sum(self.population.disease_state[self.population.node_id == i] == 2)
            for i in range(self.nodes)
        ]

    def add_component(self, component):
        """
        Adds a component (e.g., Migration, Transmission, Recovery) to the model.

        Args:
            component: An instance of a component to be added.
        """
        self.components.append(component)

    def step(self):
        """
        Advances the simulation by one timestep, updating all components and recording results.
        """
        for component in self.components:
            component.step()

        # Record results
        for i in range(self.nodes):
            in_node = self.population.node_id == i
            self.results[self.current_timestep, i, 0] = (self.population.disease_state[in_node] == 0).sum()
            self.results[self.current_timestep, i, 1] = (self.population.disease_state[in_node] == 1).sum()
            self.results[self.current_timestep, i, 2] = (self.population.disease_state[in_node] == 2).sum()

    def run(self):
        """
        Runs the simulation for the configured number of timesteps.
        """
        from tqdm import tqdm
        for self.current_timestep in tqdm(range(self.params["timesteps"])):
            self.step()

    def save_results(self, filename):
        """
        Saves the simulation results to a CSV file.

        Args:
            filename (str): Path to the output file.
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestep", "Node", "Susceptible", "Infected", "Recovered"])
            for t in range(self.params["timesteps"]):
                for node in range(self.nodes):
                    writer.writerow([t, node, *self.results[t, node]])

    def plot_results(self):
        """
        Plots the prevalence of infected agents over time for all nodes.
        """
        plt.figure(figsize=(10, 6))
        for i in range(self.nodes):
            prevalence = self.results.I[:, i] / (
                self.results.S[:, i] +
                self.results.I[:, i] +
                self.results.R[:, i]
            )
            plt.plot(prevalence, label=f"Node {i}")

        plt.title("Prevalence Across All Nodes")
        plt.xlabel("Timesteps")
        plt.ylabel("Prevalence of Infected Agents")
        plt.legend()
        plt.show()
```
///

To add migration between nodes, we will need to select the type of migration model to use and import the component. Here, we will use the sequential migration matrix to move agents sequentially between nodes. The 0th node is the 'urban' node which contains the largest population and where we seed the infection. The infection will travel sequentially from node to node, but the below example breaks the connection at node 13 for demonstrative purposes.


/// details | Code example: Adding migration using the sequential migration matrix

```
class MigrationComponent:
    """
    Handles migration behavior of agents between nodes in the model.

    Attributes:
        model (MultiNodeSIRModel): The simulation model instance.
        migration_matrix (ndarray): A matrix representing migration probabilities between nodes.
    """

    def __init__(self, model):
        """
        Initializes the MigrationComponent.

        Args:
            model (MultiNodeSIRModel): The simulation model instance.
        """
        self.model = model

        # Set initial migration timers
        max_timer = int(1 / model.params["migration_rate"])
        model.population.migration_timer[:] = np.random.randint(1, max_timer + 1, size=model.params["population_size"])

        self.migration_matrix = self.get_sequential_migration_matrix(model.nodes)

        # Example customization: Disable migration from node 13 to 14
        def break_matrix_node(matrix, from_node, to_node):
            matrix[from_node][to_node] = 0
        break_matrix_node(self.migration_matrix, 13, 14)

    def get_sequential_migration_matrix(self, nodes):
        """
        Creates a migration matrix where agents can only migrate to the next sequential node.

        Args:
            nodes (int): Number of nodes in the simulation.

        Returns:
            ndarray: A migration matrix where migration is allowed only to the next node.
        """
        migration_matrix = np.zeros((nodes, nodes))
        for i in range(nodes - 1):
            migration_matrix[i, i + 1] = 1.0
        return migration_matrix

    def step(self):
        """
        Updates the migration state of the population by determining which agents migrate
        and their destinations based on the migration matrix.
        """
        node_ids = self.model.population.node_id

        # Decrement migration timers
        self.model.population.migration_timer -= 1

        # Identify agents ready to migrate
        migrating_indices = np.where(self.model.population.migration_timer <= 0)[0]
        if migrating_indices.size == 0:
            return

        # Shuffle migrants and assign destinations based on migration matrix
        np.random.shuffle(migrating_indices)
        destinations = np.empty(len(migrating_indices), dtype=int)
        for origin in range(self.model.nodes):
            origin_mask = node_ids[migrating_indices] == origin
            num_origin_migrants = origin_mask.sum()

            if num_origin_migrants > 0:
                # Assign destinations proportionally to migration matrix
                destination_counts = np.round(self.migration_matrix[origin] * num_origin_migrants).astype(int)
                destination_counts = np.maximum(destination_counts, 0)  # Clip negative values
                if destination_counts.sum() == 0:  # No valid destinations
                    destinations[origin_mask] = origin  # Stay in the same node
                    continue
                destination_counts[origin] += num_origin_migrants - destination_counts.sum()  # Adjust rounding errors

                # Create ordered destination assignments
                destination_indices = np.repeat(np.arange(self.model.nodes), destination_counts)
                destinations[origin_mask] = destination_indices[:num_origin_migrants]

        # Update node IDs of migrants
        node_ids[migrating_indices] = destinations

        # Reset migration timers for migrated agents
        self.model.population.migration_timer[migrating_indices] = np.random.randint(
            1, int(1 / self.model.params["migration_rate"]) + 1, size=migrating_indices.size
        )
```
///

To create more complicated and more realistic migration dynamics, instead of using sequential migration we can use the gravity model to implement 2-D migration. Migration rates are proportional to population sizes, but the example still uses synthetic distances for ease of demonstration.

/// details | Code example: Adding migration using the gravity model of migration

```
class MigrationComponent2D:
    """
    Handles migration behavior of agents between nodes in the model.

    Attributes:
        model (MultiNodeSIRModel): The simulation model instance.
        migration_matrix (ndarray): A matrix representing migration probabilities between nodes.
    """

    def __init__(self, model):
        """
        Initializes the MigrationComponent.

        Args:
            model (MultiNodeSIRModel): The simulation model instance.
        """
        self.model = model

        # Set initial migration timers
        max_timer = int(1 / model.params["migration_rate"])
        model.population.migration_timer[:] = np.random.randint(1, max_timer + 1, size=model.params["population_size"])

        self.migration_matrix = self.get_gravity_migration_matrix(model.nodes)

    def get_gravity_migration_matrix(self, nodes):
        """
        Generates a gravity-based migration matrix based on population and distances between nodes.

        Args:
            nodes (int): Number of nodes in the simulation.

        Returns:
            ndarray: A migration matrix where each row represents probabilities of migration to other nodes.
        """
        pops = np.array([np.sum(self.model.population.node_id == i) for i in range(nodes)])
        distances = np.ones((nodes, nodes)) - np.eye(nodes)
        migration_matrix = gravity(pops, distances, k=1.0, a=0.5, b=0.5, c=2.0)
        migration_matrix = migration_matrix / migration_matrix.sum(axis=1, keepdims=True)
        return migration_matrix

    def step(self):

        """
        Executes the migration step for the agent-based model.

        This function selects a fraction of agents to migrate based on expired migration timers.
        It then changes their node_id according to the migration matrix, ensuring that movements
        follow the prescribed probability distributions.

        Steps:
        - Selects a subset of the population for migration.
        - Determines the origin nodes of migrating agents.
        - Uses a multinomial draw to assign new destinations based on the migration matrix.
        - Updates the agents' node assignments accordingly.

        Returns:
            None
        """
        # Decrement migration timers
        self.model.population.migration_timer -= 1

        # Identify agents ready to migrate
        migrating_indices = np.where(self.model.population.migration_timer <= 0)[0]
        if migrating_indices.size == 0:
            return

        np.random.shuffle(migrating_indices)

        origins = model.population.node_id[migrating_indices]
        origin_counts = np.bincount(origins, minlength=model.params["nodes"])

        offset = 0

        for origin in range(model.params["nodes"]):
            count = origin_counts[origin]
            if count == 0:
                continue

            origin_slice = migrating_indices[offset : offset + count]
            offset += count

            row = self.migration_matrix[origin]
            row_sum = row.sum()
            if row_sum <= 0:
                continue

            fraction_row = row / row_sum
            destination_counts = np.random.multinomial(count, fraction_row)
            destinations = np.repeat(np.arange(model.params["nodes"]), destination_counts)
            model.population.node_id[origin_slice] = destinations[:count]

        # Reset migration timers for migrated agents
        self.model.population.migration_timer[migrating_indices] = np.random.randint(
            1, int(1 / self.model.params["migration_rate"]) + 1, size=migrating_indices.size
        )
```
///


After selecting your desired migration patterns, you will need to add in a transmission component to create disease dynamics.

/// details | Code example: Adding in disease transmission

```
class TransmissionComponent:
    """
    Handles the disease transmission dynamics within the population.

    Attributes:
        model (MultiNodeSIRModel): The simulation model instance.
    """

    def __init__(self, model):
        """
        Initializes the TransmissionComponent and infects initial agents.

        Args:
            model (MultiNodeSIRModel): The simulation model instance.
        """
        self.model = model

    def step(self):
        """
        Simulates disease transmission for each node in the current timestep.
        """
        for i in range(self.model.nodes):
            in_node = self.model.population.node_id == i
            susceptible = in_node & (self.model.population.disease_state == 0)
            infected = in_node & (self.model.population.disease_state == 1)

            num_susceptible = susceptible.sum()
            num_infected = infected.sum()
            total_in_node = in_node.sum()

            if total_in_node > 0 and num_infected > 0 and num_susceptible > 0:
                infectious_fraction = num_infected / total_in_node
                susceptible_fraction = num_susceptible / total_in_node

                new_infections = int(
                    self.model.params["transmission_rate"] * infectious_fraction * susceptible_fraction * total_in_node
                )

                susceptible_indices = np.where(susceptible)[0]
                newly_infected_indices = np.random.choice(susceptible_indices, size=new_infections, replace=False)

                self.model.population.disease_state[newly_infected_indices] = 1
                self.model.population.recovery_timer[newly_infected_indices] = np.random.randint(5, 15, size=new_infections)
```
///

Finally, we need to add recovery dynamics to the model to move agents through the disease progression.

/// details | Code example: Adding in recovery dynamics

```
class RecoveryComponent:
    """
    Handles the recovery dynamics of infected individuals in the population.

    Attributes:
        model (MultiNodeSIRModel): The simulation model instance.
    """

    def __init__(self, model):
        """
        Initializes the RecoveryComponent.

        Args:
            model (MultiNodeSIRModel): The simulation model instance.
        """
        self.model = model

    def step(self):
        """
        Updates the recovery state of infected individuals, moving them to the recovered state
        if their recovery timer has elapsed.
        """
        infected = self.model.population.disease_state == 1
        self.model.population.recovery_timer[infected] -= 1
        self.model.population.disease_state[(infected) & (self.model.population.recovery_timer <= 0)] = 2
```
///

To run the created model and visualize your output, we will need to set our model parameters and run the simulation.

/// details | Code example: Running your spatial SIR model

```
# Parameters
params = {
    "population_size": 1_000_000,
    "nodes": 20,
    "timesteps": 600,
    "initial_infected_fraction": 0.01,
    "transmission_rate": 0.25,
    "migration_rate": 0.001
}

# Run simulation
model = MultiNodeSIRModel(params)
model.add_component(MigrationComponent(model))
model.add_component(TransmissionComponent(model))
model.add_component(RecoveryComponent(model))
model.run()
model.save_results("simulation_results.csv")
model.plot_results()
```
///


### Using real data

To utilize SIR models to understand real-world transmission dynamics, you will need to use real data. Model structure will be similar to what was presented above, but instead of using a synthetic population we will initialize the model using real population data. In this example, we will use data from Rwanda. You will want your data saved in a .csv file, with the following information:

- Region_id: node location, here each node is a city in Rwanda
- Population: the population of the node
- Centroid_lat and centroid-long: the latitude and longitude at the center of the node
- Birth_rate: the birth rate for the node

{{ read_csv('rwanda.csv') }}



The model code will be very similar to the code used above, but the population data will be loaded from the .csv file instead of created synthetically. In the following example, numbers are rounded and scaled down (which is optional), and each node is assigned an ID.

/// details | Code example: Creating a model using data loaded from a CSV

```
class SpatialSIRModelRealData:
    def __init__(self, params, population_data):
        """
            Initialize the mode, LASER-style, using the population_data loaded from a csv file (pandas).
            Create nodes and a migration_matrix based on populations and locations of each node.

        """
        self.params = params
        # We scale down population here from literal values but this may not be necessary.
        population_data["scaled_population"] = (population_data["population"] / params["scale_factor"]).round().astype(int)
        total_population = int(population_data["scaled_population"].sum())
        print( f"{total_population=}" )

        # Set up the properties as before
        self.population = LaserFrame(capacity=total_population, initial_count=total_population)
        self.population.add_scalar_property("node_id", dtype=np.int32)
        self.population.add_scalar_property("disease_state", dtype=np.int32, default=0)
        self.population.add_scalar_property("recovery_timer", dtype=np.int32, default=0)
        self.population.add_scalar_property("migration_timer", dtype=np.int32, default=0)

        node_pops = population_data["scaled_population"].values
        self.params["nodes"] = len(node_pops)

        node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(node_pops)])
        np.random.shuffle(node_ids)
        self.population.node_id[:total_population] = node_ids

        # seed in big node
        big_node_id = np.argmax( node_pops )
        available_population = population_data["scaled_population"][big_node_id]
        initial_infected = int(params["initial_infected_fraction"] * available_population)
        infection_indices = np.random.choice(np.where(self.population.node_id == big_node_id)[0], initial_infected, replace=False)
        self.population.disease_state[infection_indices] = 1
        self.population.recovery_timer[infection_indices] = np.random.uniform(params["recovery_time"] - 3, params["recovery_time"] + 3, size=initial_infected).astype(int)

        pop_sizes = np.array(node_pops)
        latitudes = population_data["centroid_lat"].values
        longitudes = population_data["centroid_long"].values
        distances = np.zeros((self.params["nodes"], self.params["nodes"]))

        # Nested for loop here is optimized for readbility, not performance
        for i in range(self.params["nodes"]):
            for j in range(self.params["nodes"]):
                if i != j:
                    distances[i, j] = distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

        # Set up our migration_matrix based on gravity model and input data (pops & distances)
        self.distances = distances
        self.migration_matrix = gravity(pop_sizes, distances, k=10.0, a=1.0, b=1.0, c=1.0)
        self.migration_matrix /= self.migration_matrix.sum(axis=1, keepdims=True) # normalize
```
///

### Alternative migration approach

In the above examples, we modeled migration by moving individual agents from node to node. An alternative approach is to model the migration of infection instead of individuals; this allows for computational efficiency while maintaining accurate disease transmission dynamics. Note that in this example, we do not use a `MigrationComponent` or `migration_timer`.

/// details | Code example: Modeling migration of infection instead of individuals

```
import numpy as np
from laser_core.migration import gravity
from laser_core.utils import calc_distances

class TransmissionComponent:
    """
    Transmission Component
    =======================

    This class models the transmission of disease using "infection migration"
    instead of human movement. Instead of tracking individual movement,
    infection is spread probabilistically based on a gravity-inspired network.

    This approach can significantly improve computational efficiency for
    large-scale spatial epidemic simulations.

    Attributes:
    ------------
    model : object
        The simulation model containing population and node information.
    network : ndarray
        A matrix representing the transmission probabilities between nodes.
    locations : ndarray
        Array of node latitude and longitude coordinates.
    """
    def __init__(self, model):
        """
        Initializes the transmission component by computing the infection migration network.

        Parameters:
        -----------
        model : object
            The simulation model containing population and node information.
        """
        self.model = model
        model.nodes.add_vector_property("network", length=model.nodes.count, dtype=np.float32)
        self.network = model.nodes.network

        # Extract node locations and populations from model.population_data
        self.locations = np.column_stack((model.population_data["centroid_lat"], model.population_data["centroid_long"]))
        initial_populations = np.array(model.population_data["population"])

        # Initialize heterogeneous transmission factor per agent (0.5 to 2.0)
        self.model.population.tx_hetero_factor = np.random.uniform(0.5, 2.0, size=model.population.capacity)

        # Compute the infection migration network based on node populations.
        a, b, c, k = self.model.params.a, self.model.params.b, self.model.params.c, self.model.params.k

        # Compute all pairwise distances in one call (this speeds up initialization significantly)
        # from laser_core.migration import gravity, row_normalizer
        # from laser_core.utils import calc_distances
        distances = calc_distances(self.locations[:, 0], self.locations[:, 1])
        self.network = gravity(initial_populations, distances, k, a, b, c)
        self.network /= np.power(initial_populations.sum(), c)  # Normalize
        self.network = row_normalizer(self.network, 0.01) # 0.01=max_frac

   def step(self):
        """
        Simulates disease transmission and infection migration across the network.

        New infections are determined deterministically based on contagion levels and susceptible fraction.
        """
        contagious_indices = np.where(self.model.population.disease_state == 1)[0]
        values = self.model.population.tx_hetero_factor[contagious_indices]  # Apply heterogeneity factor

        # Compute contagion levels per node
        contagion = np.bincount(
            self.model.population.node_id[contagious_indices],
            weights=values,
            minlength=self.model.nodes.count
        ).astype(np.float64)

        # Apply network-based infection movement
        transfer = (contagion * self.network).round().astype(np.float64)

        # Ensure net contagion remains positive after movement
        contagion += transfer.sum(axis=1) - transfer.sum(axis=0)
        contagion = np.maximum(contagion, 0)  # Prevent negative contagion

        # Infect susceptible individuals in each node deterministically
        for i in range(self.model.nodes.count):
            node_population = np.where(self.model.population.node_id == i)[0]
            susceptible = node_population[self.model.population.disease_state[node_population] == 0]

            if len(susceptible) > 0:
                # Compute new infections deterministically based on prevalence and susceptible fraction
                num_new_infections = int(min(len(susceptible), (
                    self.model.params.transmission_rate * contagion[i] * len(susceptible) / len(node_population)
                )))

                # Randomly select susceptible individuals for infection
                new_infected_indices = np.random.choice(susceptible, size=num_new_infections, replace=False)
                self.model.population.disease_state[new_infected_indices] = 1

                # Assign recovery timers to newly infected individuals
                self.model.population.recovery_timer[new_infected_indices] = np.random.randint(5, 15, size=num_new_infections)

        # TODO: Potential Performance Improvement: Consider using a sparse representation for `network`
        # if many connections have very low probability. This would speed up matrix multiplications significantly.

        # TODO: Investigate parallelization of contagion computation for large-scale simulations
        # using Numba or JIT compilation to optimize the loop structure.
```
///
