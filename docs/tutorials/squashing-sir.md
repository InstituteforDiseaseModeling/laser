# Add squashing and snapshot support to SIR models

The previous example demonstrated how to build SIR models, with and without spatial structure. As your models become more complex, utilizing methods to improve performance becomes important. Here, we will demonstrate how to implement squashing and snapshot support in your model.

This tutorial will demonstrate:

- Agent squashing based on recovery state
- Pre-squash result capture
- Snapshot saving and loading
- Node-level time series tracking
- Plotting of total S, I, and R dynamics

```
import numpy as np
import click
import matplotlib.pyplot as plt
from pathlib import Path

from laser_core import LaserFrame, PropertySet

class Transmission:
    """
    A simple transmission component that spreads infection within each node.
    """
    def __init__(self, population, pars):
        self.population = population
        self.pars = pars

    def step(self):
    """
    For each node in the population, calculate the number of new infections as a function of:
    - the number of infected individuals,
    - the number of susceptibles,
    - adjustments for migration and seasonality,
    - and individual-level heterogeneity.

    Then, select new infections at random from among the susceptible individuals in each node,
    and initiate infection in those individuals.
    """
    pass  # Implementation omitted for documentation purposes

    @classmethod
    def init_from_file(cls, population, pars):
        return cls(population, pars)

class Progression:
    """
    A simple progression component that recovers infected individuals probabilistically.
    """
    def __init__(self, population, pars):
        self.population = population
        self.pars = pars

    def step(self):
    """
    At each time step, update the disease state of infected individuals based on the model's
    progression logic. This may be driven by probabilities, timers, or other intrahost dynamics.
    """
    pass  # Implementation omitted for documentation

    @classmethod
    def init_from_file(cls, population, pars):
        return cls(population, pars)

class RecoveredSquashModel:
    """
    A simple multi-node SIR model demonstrating use of LASER's squash and snapshot mechanisms.
    """
    def __init__(self, num_agents=100000, num_nodes=20, timesteps=365):
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.population = LaserFrame(capacity=num_agents, initial_count=num_agents)
        self.population.add_scalar_property("node_id", dtype=np.int32)
        self.population.add_scalar_property("disease_state", dtype=np.int8)  # 0=S, 1=I, 2=R

        self.results = LaserFrame(capacity=self.num_nodes)
        self.results.add_vector_property("S", length=timesteps, dtype=np.int32)
        self.results.add_vector_property("I", length=timesteps, dtype=np.int32)
        self.results.add_vector_property("R", length=timesteps, dtype=np.int32)

        self.pars = PropertySet({
            "r0": 2.5,
            "migration_k": 0.1,
            "seasonal_factor": 0.8,
            "transmission_prob": 0.2,
            "recovery_days": 14
        })

        self.components = [
            Transmission(self.population, self.pars),
            Progression(self.population, self.pars)
            # could add other components like vaccination
        ]

    def initialize(self):
        np.random.seed(42)
        self.population.node_id[:] = np.random.randint(0, self.num_nodes, size=self.num_agents)
        recovered = np.random.rand(self.num_agents) < 0.6
        self.population.disease_state[:] = np.where(recovered, 2, 0)

    def seed_infections(self):
        susceptible = self.population.disease_state == 0
        num_seed = max(1, int(0.001 * self.population.count))
        seed_indices = np.random.choice(np.where(susceptible)[0], size=num_seed, replace=False)
        self.population.disease_state[seed_indices] = 1

    def squash_recovered(self):
        """
        Removes all agents who are recovered (state 2).
        This reduces memory footprint and speeds up simulation.
        """
        keep = self.population.disease_state[:self.population.count] != 2
        self.population.squash(keep)

    def populate_results(self):
        """
        Populate initial R values before squashing to reflect the pre-squash immunity landscape.
        """
        for nid in range(self.num_nodes):
            initial_r = ((self.population.disease_state == 2) & (self.population.node_id == nid)).sum()
            decay = np.linspace(initial_r, initial_r * 0.9, self.timesteps, dtype=int)
            self.results.R[:, nid] = decay
        print("Initial R counts per node:", self.results.R[0, :])
        print("Total initial R (summed):", self.results.R[0, :].sum())

    def run(self):
        for t in range(self.timesteps):
            for component in self.components:
                component.step()
            for nid in range(self.num_nodes):
                self.results.S[t, nid] = ((self.population.node_id == nid) & (self.population.disease_state == 0)).sum()
                self.results.I[t, nid] = ((self.population.node_id == nid) & (self.population.disease_state == 1)).sum()
                self.results.R[t, nid] += ((self.population.node_id == nid) & (self.population.disease_state == 2)).sum()

    def save(self, path):
        """
        Save the current model state to an HDF5 file, including population frame,
        pre-squash results, and simulation parameters.
        """
        self.population.save_snapshot(path, results_r=self.results.R, pars=self.pars)

    @classmethod
    def load(cls, path):
        """
        Reload a model from an HDF5 snapshot. Note: reloaded population will have
        only post-squash agents (e.g., susceptibles and infected).
        """
        pop, results_r, pars = LaserFrame.load_snapshot(path)
        model = cls(num_agents=pop.capacity, num_nodes=results_r.shape[1], timesteps=results_r.shape[0])
        model.population = pop
        model.results.R[:, :] = results_r
        model.pars = PropertySet(pars)
        model.pars["transmission_prob"] /= 10  # example modification after reload
        model.components = [
            Transmission.init_from_file(model.population, model.pars),
            Progression.init_from_file(model.population, model.pars)
        ]
        return model

    def plot(self):
        """
        Plot the time series of total S, I, and R across all nodes.
        """
        # details omitted

@click.command()
@click.option("--init-pop-file", type=click.Path(), default=None, help="Path to snapshot to resume from.")
@click.option("--output", type=click.Path(), default="model_output.h5")
def main(init_pop_file, output):
    if init_pop_file:
        model = RecoveredSquashModel.load(init_pop_file)
        model.run()
        model.plot()
    else:
        model = RecoveredSquashModel()
        model.initialize()
        model.seed_infections()
        model.populate_results()
        model.squash_recovered()
        model.save(output)
        print(f"Initial population saved to {output}")

if __name__ == "__main__":
    main()
```
