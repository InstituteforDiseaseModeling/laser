# LASER documentation

LASER (Light Agent Spatial modeling for ERadication) is a high-performance, agent-based simulation framework for modeling the spread of infectious diseases. It supports spatial structure, age demographics, and modular disease logic using Python-based components.

The LASER framework is designed to be flexible. The basis of the framework, `laser-core`, is composed of modular components that can be used to create custom epidemiological models. For those who want to explore disease dynamics without the need to code from scratch, the development team is creating [pre-built models](get-started/index.md), which will include a generic epidemiological model and disease-specific models. These pre-built models range from simple compartmental models to more complex agent-based models with spatial dynamics. And finally, for those who want to [contribute to code](contributions.md), the framework is open source and contributions are welcome!

## Learn more

<div class="grid cards" markdown>

-   :material-laptop:{ .lg .middle } __Get started modeling__

    ---

    Create and run simulations.

    [:octicons-arrow-right-24: Get started modeling](get-started/index.md)

-   :material-api:{ .lg .middle } __Reference__

    ---

    Full details on all classes and functions.

    [:octicons-arrow-right-24: API reference](parameters.md)

-   :simple-jupyter:{ .lg .middle } __Tutorials__

    ---

    An interactive tour of key features.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :octicons-book-16:{ .lg .middle } __Glossary__

    ---

    Look up unfamiliar terms.

    [:octicons-arrow-right-24: Glossary](glossary.md)


-   :material-new-box:{ .lg .middle } __What's new__

    ---

    See what's in the latest releases.

    [:octicons-arrow-right-24: What's new](whatsnew.md)


</div>

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

SO: the docs are going to need to have instructions and help on how to do all of these. I've added notes on where the info should go. As mentioned, don't split these up into persona buckets, just make sure the tasks are explained in order of start - finish (building up complexity).

We can change what the buttons link to, these are sort of placeholders right now.-->


{%
    include-markdown "../AUTHORS.md"
%}
