===============
SIR Model Example
===============

This document provides a comprehensive example of a single-node Susceptible-Infected-Recovered (SIR) model implemented using the `LASERFrame` and `PropertySet` libraries. This example demonstrates how to structure a disease transmission model, include components for transmission and recovery, and visualize results. The example highlights features such as reporting and plotting for model evaluation.

Introduction
------------
The SIR model presented here simulates disease dynamics within a closed population using the `LASERFrame` framework. The population starts with a defined number of susceptible and infected individuals, progresses over time with recovery and transmission components, and tracks results for visualization. This example serves as a practical guide for modeling simple epidemic dynamics.

Code Implementation
--------------------

Model Class
~~~~~~~~~~~

The `SIRModel` class is the core of the implementation. It initializes a population using `LaserFrame`, sets up disease state and recovery timer properties, and tracks results across timesteps.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from laser_core.laserframe import LaserFrame
    from laser_core.propertyset import PropertySet

    class SIRModel:
        def __init__(self, params):
            self.population = LaserFrame(capacity=params.population_size)
            self.population.add_scalar_property("disease_state", dtype=np.int32, default=0)
            self.population.add_scalar_property("recovery_timer", dtype=np.int32, default=0)
            self.population.add(params.population_size)
            self.population.disease_state[:] = 0
            self.population.disease_state[0:10] = 1
            self.params = params
            self.results = {"S": [], "I": [], "R": []}
            self.components = []

        def add_component(self, component):
            self.components.append(component)

        def track_results(self):
            susceptible = (self.population.disease_state == 0).sum()
            infected = (self.population.disease_state == 1).sum()
            recovered = (self.population.disease_state == 2).sum()
            total = len(self.population)
            self.results["S"].append(susceptible / total)
            self.results["I"].append(infected / total)
            self.results["R"].append(recovered / total)

        def run(self):
            for _ in range(self.params.timesteps):
                for component in self.components:
                    component.step()
                self.track_results()

        def plot_results(self):
            plt.figure(figsize=(10, 6))
            plt.plot(self.results["S"], label="Susceptible (S)", color="blue")
            plt.plot(self.results["I"], label="Infected (I)", color="red")
            plt.plot(self.results["R"], label="Recovered (R)", color="green")
            plt.title("SIR Model Dynamics with LASER Components")
            plt.xlabel("Time (Timesteps)")
            plt.ylabel("Fraction of Population")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig("gpt_sir.png")

Intrahost Progression Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `IntrahostProgression` class manages recovery dynamics by updating infected individuals based on a given recovery rate.

.. code-block:: python

    class IntrahostProgression:
        def __init__(self, model):
            self.population = model.population
            self.recovery_rate = model.params.recovery_rate

        def step(self):
            infected = self.population.disease_state == 1
            recoveries = np.random.rand(infected.sum()) < self.recovery_rate
            self.population.disease_state[infected] = np.where(recoveries, 2, 1)

Transmission Component
~~~~~~~~~~~~~~~~~~~~~~~

The `Transmission` class manages disease spread by modeling interactions between susceptible and infected individuals.

.. code-block:: python

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
            infection_probability = self.infection_rate * (num_infected / population_size)
            new_infections = np.random.rand(num_susceptible) < infection_probability
            susceptible_indices = np.where(susceptible)[0]
            new_infected_indices = susceptible_indices[new_infections]
            self.population.disease_state[new_infected_indices] = 1

Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~~

The simulation parameters are defined using the `PropertySet` class.

.. code-block:: python

    params = PropertySet({
        "population_size": 100_000,
        "infection_rate": 0.3,
        "recovery_rate": 0.1,
        "timesteps": 160
    })

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~

The model is initialized with the defined parameters, components are added, and the simulation is run for the specified timesteps. Results are then visualized.

.. code-block:: python

    sir_model = SIRModel(params)
    sir_model.add_component(IntrahostProgression(sir_model))
    sir_model.add_component(Transmission(sir_model))
    sir_model.run()
    sir_model.plot_results()

Conclusion
----------

This example demonstrates a robust implementation of a single-node SIR model using `LASERFrame` and `PropertySet`. It showcases modular design, efficient result tracking, and intuitive visualization of epidemic dynamics. This example can be extended with features like vaccination or age-structured populations for advanced modeling.
