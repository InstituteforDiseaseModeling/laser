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

## Technicalities

|Minimal Payload|bytes|
|---------------|:---:|
| immune status (T/F) | 1 |
| infectious timer (\< 256 days) | 1 |
| &Sigma; | **2** |

|Realistic Payload|bytes|
|-----------------|:---:|
| age of DoB (115 years = 41,975 days) | 2 |
| susceptibility (would use float16 on GPU) | 4 |
| accessibility | 1 |
| SES/nutrition status | 1 |
| exposed timer (\< 256 days) | 1 |
| infectious timer (\< 256 days) | 1 |
| age at infection | 2 |
| community ID (up to 65K communities) | 2 |
| &Sigma; | **14** |

|Population|Minimal Payload<br>(2 bytes)|Minimally Realistic<br>Payload (14 bytes)|CPAS*|pass/sec†|notes|
|:--------:|:---------------------:|:---------------:|:-:|:-:|:-:|
|1M  (10<sup>6</sup>)|2MB|14MB|2,000|23,000/~3,300||
|10M (10<sup>7</sup>)|20MB|140MB|200|2,300/~330|megacity|
|100M (10<sup>8</sup>)|200MB|1.4GB|20|230/~33||
|200M (2x10<sup>8</sup>)|400MB|2.8GB|10|115/~16|Nigeria (231M)<br>Pakistan (242M)|
|1G (10<sup>9</sup>)|2GB|14GB|2|23/~3.3||

*CPAS - cycles/agent/second (2GHz CPU). SIMD instructions could be a multiplier on the number of instructions/cycle, e.g. AVX on Intel with 256-bit registers can operate on 8 float32s simultaneously. Apple Silicon supports ARM NEON with 128-bit registers.

†How many passes over memory/second based on memory bandwidth (Lenovo ~46GB/s, Apple Silicon up to 800GB/s).

### PRNG State Considerations

PRNG state appears to use between 16 and 268 bytes depending on the algorithm. CPU based modeling can probably allocate state for each CPU code in use (vanishingly small memory use). GPU based modeling will probably need to group agents to use draws from a shared PRNG to amortize the memory usage across several agents. E.g., 16 agents using the same PRNG with 16 bytes of state only incur +1 byte of state/agent.

|Algorithm|State Size (bytes)|
|:--------|:----------------:|
|XOROSHIRO128+| 16 |
|other XOROSHIRO variations| 16+ |
|XORWOW| 44 |
|MRG32k3a| 44 |
|Sobel32| 140 |
|Sobel64| 268 |
