# LASER documentation

LASER (Light Agent Spatial modeling for ERadication) is a high-performance, agent-based simulation framework for modeling the spread of infectious diseases. It supports spatial structure, age demographics, and modular disease logic using Python-based components. LASER provides flexible components for creating, extending, and calibrating dynamic epidemiological models.

LASER has several pre-built models that you can use to explore disease dynamics without the need to code a model from scratch. These models include built-in examples ranging from simple compartmental models to more complex agent-based models with spatial dynamics. The modular structure of LASER also enables you to build custom models to fit particular research needs, through the integration or modification of the LASER-core components. As an open source model, the LASER team also encourages experienced users to extend the LASER framework by contributing to code!


<!-- [Don't write out personas or split tasks into persona groups; the docs should be task-oriented, so users can determine what they need by what tasks they're trying to accomplish. Understanding personas is an internal tool so we can appropriately identify tasks & necessary info]


As a reminder, the following tasks were listed in the original intro:

- Run powerful simulations of disease dynamics without building models from scratch. [Running sims]
- Leverage built-in examples for SIR, vital dynamics, spatial modeling, and calibration. [Code snippets in demographics, calibration; how-to in running sims, adding spatial dynamics, and tutorials]
- Gain insights into how spatial spread, birth/death, or vaccination influence transmission. [Tutorials]
- Run calibrations against real-world data to optimize model parameters. [Calibration]
- Compose custom models by integrating or modifying modular components, such as transmission, immunity, and migration. [getting started/running sims/custom models]
- Add epidemiologically relevant features like contact tracing or waning immunity. [Running sims]
- Run calibrations against real-world data to optimize model parameters. [calibration]
- Extend the LASER framework with new core functionality: algorithms, optimization backends, spatial logic. [development]
- Contribute performance-critical modules using Numba, OpenMP, or C. [development]

SO: the docs are going to need to have instructions and help on how to do all of these. I've added notes on where the info should go. As mentioned, don't split these up into persona buckets, just make sure the tasks are explained in order of start - finish (building up complexity). -->