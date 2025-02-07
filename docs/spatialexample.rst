Simple Spatial SIR Model with Synthetic Data
============================================

This example simulates a spatial Susceptible-Infected-Recovered (SIR) model with modular components. The population is distributed across 20 nodes in a 1-D chain, with infection spreading spatially from node 0 and agents migrating based on a migration matrix.

Spatial Arrangement
-------------------

The model uses a 1-D spatial structure:
- **Node connections**: Each node is connected to its next neighbor in the chain, allowing migration to occur sequentially (e.g., 0 → 1 → 2 ...).
- **Infection propagation**: Infection starts in node 0 and spreads to neighboring nodes as agents migrate.

Two migration matrix options are available:
1. **Sequential Migration Matrix**: Agents can only move to their next node in the chain.
2. **Gravity Model Migration Matrix**: This provides a more realistic 2-D spatial dynamic, where migration probabilities depend on node distances and population sizes.

Population Initialization
--------------------------

- **Skewed Distribution**: The population is distributed across nodes using a rural-to-urban skew.
- **Timers**: Migration timers are assigned to control agent migration frequency.

Model Components
----------------

The simulation uses modular components for migration, transmission, and recovery dynamics. Each component encapsulates the logic for its specific task.

Full Code Implementation
-------------------------

Imports
^^^^^^^

.. code-block:: python
    :linenos:

    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    from laser_core.laserframe import LaserFrame
    from laser_core.demographics.spatialpops import distribute_population_skewed as dps
    from laser_core.migration import gravity


Model Class
^^^^^^^^^^^

.. code-block:: python
    :linenos:

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
            self.results.S[self.current_timestep, :] = np.array([
                np.sum(self.population.disease_state[self.population.node_id == i] == 0)
                for i in range(self.nodes)
            ])
            self.results.I[self.current_timestep, :] = np.array([
                np.sum(self.population.disease_state[self.population.node_id == i] == 1)
                for i in range(self.nodes)
            ])
            self.results.R[self.current_timestep, :] = np.array([
                np.sum(self.population.disease_state[self.population.node_id == i] == 2)
                for i in range(self.nodes)
            ])

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


Migration Component Class
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

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


Transmission Component Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

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

Recovery Component Class
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

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


Run Everything
^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

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

Discussion
^^^^^^^^^^

This simple spatial example with migration creates a set of nodes with synthetic populations and a linear connection structure. The 0th node is the 'urban' node, with the largest population, and where we seed the infection. The migration matrix just connects nodes to the next node (by index). So we expect to see infection travel sequentially from node to node. We break the connection at node 13 just to show we can.

Now we make things a bit more interesting by taking a similar set of nodes, but creating a migration matrix from the gravtiy function, so we effectively have a 2D network with rates proportional to population sizes. Distances are still very synthetic. We change our migration function as well since we have to be a little smarter when our matrix isn't sparse.

Migration Component (2D)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:

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

Discussion
^^^^^^^^^^

This example is more advanced than our first one since it moves from 1 dimension to 2, and is fully connected. We should see infection spread from the large seed node to most or potentially all of the other nodes (depending on transmission rates and migration rates and stochasticity) in a way that is broadly a function of the other nodes' populations. Though since the smaller nodes don't vary massively in population and since the model is stochastic, it will be a general correlation.

Now we make things a bit more interesting by taking a similar set of nodes, but creating a migration matrix from the gravtiy function, so we effectively have a 2D network with rates proportional to population sizes. Distances are still very synthetic. We change our migration function as well since we have to be a little smarter when our matrix isn't sparse.


Simple Spatial SIR Model with Real Data
=======================================

Design
------
Now we switch from synthetic population and spatial data to real population data. In this example, we have a csv
file which starts off like this:

::

  region_id,population,centroid_lat,centroid_long,birth_rate
  Ryansoro,46482.66796875,-3.707268618580818,29.79879895068512,11.65647
  Ndava,72979.296875,-3.391556716979041,29.753430749757815,15.881549
  Buyengero,76468.8125,-3.8487418774123014,29.53299692786253,12.503805
  Bugarama,44571.8515625,-3.6904789341549504,29.400408879716224,11.025566
  Rumonge,300248.03125,-3.9622108122897663,29.45711276535364,19.567726
  Burambi,63219.703125,-3.798641437985548,29.452423323952797,9.199019
  Kanyosha1,115017.984375,-3.4309688424403473,29.41531324224386,37.951366
  Kabezi,71913.8359375,-3.5311012728218527,29.369968675926785,31.831919
  Muhuta,88141.7109375,-3.623512958448508,29.415218642943234,21.598902



Code
^^^^

.. code-block:: python
    :linenos:

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

Discussion
^^^^^^^^^^

We load the population data from the csv file, round and scale down (optional), and assign node ids. The shuffling is probably optional, but helps avoid biasing. We exploit the distance function from laser_utils (import not in code snippet).
