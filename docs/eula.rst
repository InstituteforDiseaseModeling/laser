Population Initialization, Squashing, and Snapshot Management in LASER
=======================================================================


As population models in LASER grow to very large sizes, it becomes computationally expensive
to repeatedly run sophisticated initialization routines. In many cases—particularly during
model calibration—it is far more efficient to initialize the population once, save it, and
then reload the initialized state for subsequent runs.


This approach is especially useful when working with *lightweight agents* that are
epidemiologically uninteresting—for example, agents who are already recovered or immune
in a measles or polio model. In such models, the majority of the initial population may be
in the "Recovered" state, potentially comprising 90% or more of all agents. If you are
simulating 100 million agents, storing all of them can result in excessive memory usage.


To address this, LASER supports a **squashing** process. Squashing involves
*defragmenting the data frame* such that all epidemiologically active or "interesting" agents
(e.g., Susceptible or Infectious) are moved to the beginning of the array or table, and
less relevant agents (e.g., Recovered) are moved to the end.


Following squashing:

- The population count is adjusted so that all `for` loops and step functions iterate
  **only over the active population**.
- This not only reduces memory usage but also improves performance by avoiding unnecessary
  computation over inactive agents.

Furthermore, when saving a **Snapshot**, only the active (unsquashed) portion of the
population is saved. Upon reloading:

- Only this subset is allocated in memory.
- This prevents the performance penalty of managing large volumes of unused agent data.

Important Detail
----------------

Before squashing, LASER allows you to **count and record** the number of recovered
(or otherwise squashed) agents. This count should be stored in a summary variable—
typically the ``R`` column of the results data frame. This ensures your model retains a
complete epidemiological record even though the agents themselves are no longer instantiated.


Example
-------

A working example is provided in the documentation: a simple spatial SIR model demonstrating
how to perform population squashing, save a snapshot, and reload it properly. This example
illustrates how to:

- Initialize the full population.
- Squash and record unneeded agents.
- Save only the active subset.
- Reload efficiently for future simulations.


RecoveredSquashModel Example
============================

This example demonstrates a simple multi-node SIR model using the LASER framework, including:

- Disease transmission and progression
- Population squashing to remove recovered agents
- Snapshot save/load functionality
- Time series result plotting

.. code-block:: python

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
            susceptible = self.population.disease_state == 0
            infected = self.population.disease_state == 1
            inf_node_ids = self.population.node_id[infected]
            node_counts = np.bincount(inf_node_ids, minlength=self.population.node_id.max() + 1)

            for nid in range(len(node_counts)):
                if node_counts[nid] > 0:
                    sus_idx = (self.population.node_id == nid) & susceptible
                    new_inf = np.random.rand(sus_idx.sum()) < self.pars.transmission_prob
                    indices = np.where(sus_idx)[0][new_inf]
                    self.population.disease_state[indices] = 1

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
            infected = self.population.disease_state == 1
            recoveries = np.random.rand(infected.sum()) < (1 / self.pars.recovery_days)
            self.population.disease_state[np.where(infected)[0][recoveries]] = 2

        @classmethod
        def init_from_file(cls, population, pars):
            return cls(population, pars)

    class RecoveredSquashModel:
        """
        A simple multi-node SIR model demonstrating use of LASER's squash and snapshot mechanisms.
        """
        def __init__(self, num_agents=100000, num_nodes=20, timesteps=365):
            ...
            self.components = [
                Transmission(self.population, self.pars),
                Progression(self.population, self.pars)
            ]

        def initialize(self):
            ...

        def seed_infections(self):
            ...

        def squash_recovered(self):
            """
            Removes all agents who are recovered (state 2).
            This reduces memory footprint and speeds up simulation.
            """
            ...

        def populate_results(self):
            """
            Populate initial R values before squashing to reflect the pre-squash immunity landscape.
            """
            ...

        def run(self):
            ...

        def save(self, path):
            """
            Save the current model state to an HDF5 file, including population frame,
            pre-squash results, and simulation parameters.
            """
            ...

        @classmethod
        def load(cls, path):
            """
            Reload a model from an HDF5 snapshot. Note: reloaded population will have
            only post-squash agents (e.g., susceptibles and infected).
            """
            ...

        def plot(self):
            """
            Plot the time series of total S, I, and R across all nodes.
            """
            ...

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
