<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/10873335/283954768-97685a6d-7b86-4bba-a3e6-07ac22d5a2b3.png" alt="LASER logo" width="600px"/>
</p>

## Temporary README for `cleanup-for-merge` Branch

- use `python3 -m pip install -e .` in the root to install the code including the NumPy+Numba and Taichi (GPU) implementations of spatial SEIR.
- in the `tests` directory there are command line scripts to run the two implementations:
  - `run_numpyba.py`
  - `run_taichi.py`
- in the root directory there is a notebook, `start.ipynb`, which
  - sets meta parameters
  - sets disease parameters
  - sets network parameters
  - chooses a model implementation
  - runs the model
  - plots SEIR trajectories for a node
  - plots a trajectory of %I vs. %S over time
- note: the first time you run a cell in a Jupyter notebook in a Codespace, you will get some questions from VSCode in the command bar at the top of the window -
  - "Select Kernel" &rarr; "Install/Enable suggested extensions Python + Jupyter"
  - "Select Another Kernel" &rarr; "Python Environment" &rarr; "Python 3.10.13 ~/python/current/bin/python3" <br> (the version on the last one may change over time)

----

## Schedule

### First 30 Days (EOY 2023)

- <strike>firm up team/stakeholders/advisory committee</strike> ✅
- enumerate necessary features for reproducing/supporting previous and in-progress modeling efforts
  - <strike>measles (kmccarthy)</strike> ✅
  - malaria (cbever/pselvaraj)
  - <strike>end-game/end-stage polio</strike> ✅
- <strike>enumerate necessary features for outstanding questions and issues</strike> ✅

### First 60 Days (January 31, 2024)

- <strike>"paper search" / investigate potential existing solutions</strike> ✅
- capture development requirements
  - tools for preparing data (demographics, networks, etc.)
  - file formats
  - select initial features
    - spatial connectivity
      - individual agent migration (genetics - vector _and_ parasite)
      - NxN matrix connectivity, contagion transport
      - multi-level (meso-scale?) connectivity (communities of communities)
    - community transmission dynamics
      - agents
      - cohorts
      - <strike>\*Sim</strike>
      - <strike>stochastic compartmental</strike>
      - <strike>ODEs</strike>
      - <strike>emulator</strike>
    - demographics
      - urban/rural
      - class/caste
    - multiple independent populations/community (people + mosquitoes, people + dogs, etc.)
    - ¿co-transmission? TB _and_ HIV
    - non-disease vital dynamics
  - visualization choices

### First 120 Days (February 29, 2024)

- <strike>technical considerations</strike> ✅
  - <strike>single laptop</strike>
  - <strike>single laptop w/Nvidia GPU</strike>
  - <strike>multicore</strike>
    - <strike>single machine</strike>
    - <strike>large machine (cloud)</strike>
    - <strike>beyond?</strike>
  - <strike>Numpy</strike>
  - <strike>NumPy + Numba</strike>
  - <strike>NumPy + Numba + CUDA</strike>
- &rarr; best available implementation for hardware at hand:
  - NumPy + Numba
  - SSE/AVX2/AVX512
  - OpenMP
  - CUDA (Nvidia)/Metal (Apple)

## Problem Space

The problem is inherently an issue of heterogeneity. Spatial decomposition is the easiest consideration, but not sufficient - a model of N "independent" but identical communities is generally not useful.

Spatial connectivity and the associated latencies in transmission address one dimension of heterogeneity: how "close" is a given community to a potential source of imported contagion (exogenous to the model "world", locally endogenous, e.g., an adjacent community, endogenous but at a remove - rare transmission or multi-stop chain of transmission).

Community size in a spatial model is also a consideration - what is the configuration and connectivity of sub-CCS nodes to nodes at or above CCS for the given disease?

We need configurable characteristics of the individual communities which can vary, along with their interconnectedness, to capture additional heterogeneity.

What _is_ the modeling of the individual communities? "Light-Agent" seems to limit us to an ABM, but we should consider cohorts of epidemiologically similar populations (polio >5, HIV <15, TB latents, etc.) as well as stochastic compartmental models.

Are the individual communities well-mixed or should we also provide for explicit networks at the local level?

## Technology

- Python
- NumPy / NUMBA / ¿PyCUDA?
- native code
  - C++ (somewhat awkward interop with Python, but potentially accessible from other technologies, e.g., R)
  - Rust ([PyO3](https://github.com/PyO3/pyo3) is quite nice, but requires getting up to speed on Rust 😳)
- compute requirements:
  - laptop 2010+? (might inform SIMD capabilities)
  - GPU (CUDA) enabled machine laptop/desktop/cloud
  - single core/multi-core
  - largest scenarios?
- visualization
  - cross-platform
  - real-time
- existing file formats for input data
- existing file formats for output data (GeoTIFF? - works with ArcGIS?)

## Other

- community builder tool for given total population and community size distribution
- network builder given a set of communities (gravity, radiation, other algorithms in existing libraries/packages)
- independent populations w/in a community, e.g., mosquitoes or dogs along with humans
- independent or co-transmission, i.e. multiple "diseases"
- models need to be connected with real-world scenarios, not [just] hypothetical explorations

## Notes

- "light" : How light is "light"? &rarr; # agents * state/agent <= available RAM
- "agent" : Cohorts? Stochastic compartmental? &rarr; individual agents
- "spatial" : How good are the individual community models? Good enough for non-spatial questions?
- dynamic properties (e.g. GPU flu simulation) &rarr; ✅
- ¿Ace/clorton-based state machines?
