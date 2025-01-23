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

            # Reporting structure
            self.results = np.zeros((params["timesteps"], self.nodes, 3))  # S, I, R per node

            # Components
            self.components = []

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
                prevalence = self.results[:, i, 1] / self.results[:, i, :].sum(axis=1)
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
