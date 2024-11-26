==================================
Software Architecture & Design
==================================

Summary
=======
The disease model consists of a population of agents, each of which has a set of variables and parameters that represent their health status. These variables include things like infection timers, susceptibility timers, and immunity levels. The model also includes components that determine the progression of disease within the population, such as transmission and recovery rates. The model is run over a series of time steps, with the agents interacting with each other and the environment, and their health statuses changing based on the rules defined by the model. At the end of each time step, the model may also generate output, such as reports or visualizations, to help understand the state of the population.

Layout
======

The codebase consists of:

- A top-level script, or application layer, called ``cholera.py``. 
- A Model, which is initialized from some geospatial populations. Each patch or node will start with some number of humans/agents.
- A set of agent properties which you define and add to the model.
- A set of components which are sometimes called step functions or phase functions. These update one or more properties, such as age or infection timers, each timestep.

Application Layer
=================

The main script has 5 main components:

1. Gather configuration parameters from user via command-line, ``config.json``, etc.
2. Create Model object from population and other demographic input files.
3. Load and initialize all components (phase or step functions).
4. Run main loop, invoking all step functions once each timestep for all timesteps.
5. Write out final reports, visualize, and analyze.

The Model Object
================
The main Model object contains your agent population in a ``LASERFrame`` (similar to a dataframe, but with some extra capabilities) and your patch/node 'population'. Your agent population is essentially a giant dataframe, where the columns are the properties of each agent, defined by you, and the rows are each agent. You will initialize all your values and update them each timestep as needed.

Components
==========

Input Files
===========
To run this geospatial cholera model, you need to provide input files for:

- The manifest
- Immunity
- Age distribution
- Population demographics

These files can be specified in the model's ``params`` variable. Additionally, parameters such as ``r_naught`` and ``beta`` should also be provided in ``params``.

Output Files
============

Demographics
============

- **Births**  
  TBD

- **Deaths**  
  TBD

- **Aging**  
  TBD

User Customizability
====================

1. **Configuration**

   Based on the context, the configuration parameters that need to be set are:

   - ``ticks``: the number of ticks in the simulation (int)
   - ``output``: the path to the output directory (Path)
   - ``prevalence``: the initial prevalence of the disease in the population (float)
   - ``a``: the population 1 power factor (np.float32)
   - ``b``: the population 2 power factor (np.float32)
   - ``c``: the distance power factor (np.float32)
   - ``k``: the gravity constant (np.float32)
   - ``max_frac``: the maximum fraction of population that can move in a single tick (np.float32)
   - ``exp_mean``: the expected duration of the exposed state (np.float32)
   - ``exp_std``: the standard deviation of the exposed state (np.float32)
   - ``inf_mean``: the expected duration of the infectious state (np.float32)
   - ``inf_std``: the standard deviation of the infectious state (np.float32)
   - ``r_naught``: the basic reproduction number of the disease (np.float32)
   - ``seasonality_factor``: the factor that accounts for seasonal variation in transmission (np.float32)
   - ``seasonality_phase``: the phase of the seasonal variation

2. **Input Files**

3. **Code**

New Modeler Workflow
====================

Here’s how you should break down your modeling problem to model a disease with LASER:

1. TBD
2. TBD
3. TBD

Glossary of Terms
=================
- **Patch**  
  Something...

