========
Overview
========

LASER (Light Agent Spatial modeling for ERadication) is a high-performance, agent-based simulation
framework for modeling the spread of infectious diseases. It supports spatial structure,
age demographics, and modular disease logic using Python-based components. LASER provides flexible
components for creating, extending, and calibrating dynamic epidemiological models.

LASER is designed to support the following personas: research scientist,
research scientist-developer, software engineer. The following lists some of the reasons
why you may want to use LASER:

Research scientist
------------------

* Run powerful simulations of disease dynamics without building models from scratch.
* Leverage built-in examples for SIR, vital dynamics, spatial modeling, and calibration.
* Gain insights into how spatial spread, birth/death, or vaccination influence transmission.
* Run calibrations against real-world data to optimize model parameters.

Research scientist-developer
----------------------------

* Compose custom models by integrating or modifying modular components, such as transmission, immunity, and migration.
* Add epidemiologically relevant features like contact tracing or waning immunity.
* Run calibrations against real-world data to optimize model parameters.

Software engineer
-----------------

* Extend the LASER framework with new core functionality: algorithms, optimization backends, spatial logic.
* Contribute performance-critical modules using Numba, OpenMP, or C.

To get started and for more information about each of these persona types, related tasks,
and where to find related documentation, see :doc:`gettingstarted`.