# LASER Motivation and Priorities

## Goals

* Support elimination and eradication programs through answering questions related to spatial transmission of disease.
  * Measles
  * Polio
  * Malaria
* Enable IDM researchers to lead the way with cutting edge capabilities.
  * Implement new features and capabilities previously unavailable.
* Enable new scenarios for existing models
  * Increased performance – explore larger parameter space
  * Increased performance – explore larger population scenarios on existing hardware
* Enable IDM researchers to model large scale scenarios.
  * GPU based compute
  * Multi-core/multi-threaded compute
  * Cluster based compute
* Enable external partners and users to model relevant scenarios on readily available hardware.
  * Maximize performance on commonly available hardware (laptops <= 10 years old).
  * Performance on laptops w/Nvidia GPUs.
  * Consider performance on Apple Silicon devices.
* Design for quick prototyping and extensibility.
  * Enable internal investigation into new features and capabilities
  * Enable timely development of necessary features
  * Enable external partners and users to extend and customize tools for their scenarios

## Technology

* Python w/NumPy
  * Numba as appropriate
  * PyCUDA as appropriate and possible
  * C++ or Rust for specific, high performance capabilties
* Agent based communities
  * Homogeneously mixed
  * Agent based communities with cohorts
  * Agent based communities with explicit networks
  * Stochastic, compartmental communities
* Flexible spatial connectivity component
  * “scoops” of contagion, i.e. “anonymous” contagion, e.g. measles
  * Traceable/trackable agents and infection, e.g. malaria vector or parasite genetics
  * Layered connectivity – local, regional, etc.

## Processes

* Documentation
* Scenarios and examples
* Test suite

## Other

* multiple independent species, e.g. humans, mosquitos, dogs, ...
* multiple independent transmission networks, intra-mural and inter-state
* event based modeling, e.g. draw for non-disease death date
* state machine based modeling

----

* toolset vs. builtin - e.g. is building network a tool in the toolset or a function in the model setup?
* relationship w/calibration
* 
