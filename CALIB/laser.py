import csv
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numba
import numpy as np

from laser_core.demographics.spatialpops import distribute_population_tapered as dpt
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity


@numba.njit(parallel=True)
def fast_reporting(node_id, disease_state, results, current_timestep, nodes):
    """
    Optimized reporting function using thread-local storage (TLS) for parallel accumulation.
    """
    num_threads = numba.get_num_threads()  # Get number of parallel threads

    # Thread-local storage: One array per thread to avoid race conditions
    local_s = np.zeros((num_threads, nodes), dtype=np.int32)
    local_i = np.zeros((num_threads, nodes), dtype=np.int32)
    local_r = np.zeros((num_threads, nodes), dtype=np.int32)

    # Parallel accumulation
    for i in numba.prange(len(node_id)):  # Loop through all agents in parallel
        thread_id = numba.get_thread_id()  # Get the current thread ID
        node = node_id[i]
        state = disease_state[i]
        if state == 0:
            local_s[thread_id, node] += 1
        elif state == 1:
            local_i[thread_id, node] += 1
        elif state == 2:
            local_r[thread_id, node] += 1

    # Merge thread-local storage into the final results
    for t in range(num_threads):
        for j in numba.prange(nodes):
            results[current_timestep, j, 0] += local_s[t, j]
            results[current_timestep, j, 1] += local_i[t, j]
            results[current_timestep, j, 2] += local_r[t, j]


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
        # node_pops = dps(params["population_size"], self.nodes, frac_rural=0.3)
        node_pops = dpt(params["population_size"], self.nodes)
        node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(node_pops)])
        np.random.shuffle(node_ids)
        self.population.node_id[: params["population_size"]] = node_ids

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

    def run(self):
        """
        Runs the simulation for the configured number of timesteps.
        """
        from tqdm import tqdm

        for t in tqdm(range(self.params["timesteps"])):
            self.current_timestep = t
            self.step()

    def save_results(self, filename):
        """
        Saves the simulation results to a CSV file.

        Args:
            filename (str): Path to the output file.
        """
        with Path(filename).open(mode="w", newline="") as file:
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
        for i in [0]:  # range(self.nodes):
            prevalence = self.results[:, i, 1] / self.results[:, i, :].sum(axis=1)
            plt.plot(prevalence, label=f"Node {i}")
        plt.title("Prevalence Across All Nodes")
        plt.xlabel("Timesteps")
        plt.ylabel("Prevalence of Infected Agents")
        plt.legend()
        plt.show()


class MigrationComponent1D:
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


class MigrationComponent2D:
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
        # Decrement migration timers
        self.model.population.migration_timer -= 1

        # Identify agents ready to migrate
        migrating_indices = np.where(self.model.population.migration_timer <= 0)[0]
        if migrating_indices.size == 0:
            return

        np.random.shuffle(migrating_indices)

        origins = self.model.population.node_id[migrating_indices]
        origin_counts = np.bincount(origins, minlength=self.model.params["nodes"])

        offset = 0

        for origin in range(self.model.params["nodes"]):
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
            destinations = np.repeat(np.arange(self.model.params["nodes"]), destination_counts)
            self.model.population.node_id[origin_slice] = destinations[:count]

        # Reset migration timers for migrated agents
        self.model.population.migration_timer[migrating_indices] = np.random.randint(
            1, int(1 / self.model.params["migration_rate"]) + 1, size=migrating_indices.size
        )


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

        # Infect initial individuals in node 0
        node_pops = np.bincount(self.model.population.node_id, minlength=self.model.nodes)
        initial_infected = int(self.model.params["initial_infected_fraction"] * node_pops[0])
        infected_indices = np.random.choice(np.where(self.model.population.node_id == 0)[0], size=initial_infected, replace=False)
        self.model.population.disease_state[infected_indices] = 1
        self.model.population.recovery_timer[infected_indices] = np.random.randint(5, 15, size=initial_infected)

    def step(self):
        """
        Simulates disease transmission for each node in the current timestep.
        """
        last_timestep = self.model.current_timestep  # Last recorded timestep
        results = self.model.results  # Shortcut for clarity

        fast_reporting(
            self.model.population.node_id,
            self.model.population.disease_state,
            self.model.results,
            self.model.current_timestep,
            self.model.nodes,
        )

        for i in range(self.model.nodes):
            # Use precomputed values instead of recalculating
            in_node = self.model.population.node_id == i
            susceptible = in_node & (self.model.population.disease_state == 0)
            num_infected = results[last_timestep, i, 1]
            num_susceptible = int(susceptible.sum())
            total_in_node = num_susceptible + num_infected + results[last_timestep, i, 2]

            if total_in_node > 0 and num_infected > 0 and num_susceptible > 0:
                infectious_fraction = num_infected / total_in_node
                susceptible_fraction = num_susceptible / total_in_node

                new_infections = int(self.model.params["transmission_rate"] * infectious_fraction * susceptible_fraction * total_in_node)

                susceptible_indices = np.where(susceptible)[0]
                newly_infected_indices = np.random.choice(susceptible_indices, size=new_infections, replace=False)

                self.model.population.disease_state[newly_infected_indices] = 1
                self.model.population.recovery_timer[newly_infected_indices] = np.random.randint(5, 15, size=new_infections)


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


"""
# Parameters
params = {
    "population_size": 500_000,
    "nodes": 20,
    "timesteps": 300,
    "initial_infected_fraction": 0.01,
    "transmission_rate": 0.15,
    "migration_rate": 0.005
}
"""


@click.command()
@click.option("-p", "--params", default="params.json", help="Path to parameters JSON file")
@click.option("-o", "--output", default="simulation_results.csv", help="Output filename for simulation results")
@click.option("--plot", is_flag=True, help="Enable plotting of results")
def main(params, output, plot):
    # We expect some part of optuna to set the params.json as a prior step
    with Path(params).open("r") as par:
        params = json.load(par)

    # Run simulation
    model = MultiNodeSIRModel(params)
    model.add_component(TransmissionComponent(model))
    model.add_component(RecoveryComponent(model))
    model.add_component(MigrationComponent2D(model))
    model.run()

    # Save raw results for optuna to analyze; this might actually be annoyingly slow across a full calibration. But it's simple
    # One can either store hdf5 or do the post-proc 'analyzer' step first and just save that.
    print(f"DEBUG Saving {output}")
    model.save_results(output)

    # Don't want to plot
    if plot:
        model.plot_results()


if __name__ == "__main__":
    main()
