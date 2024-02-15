# Proof-of-Concept Models Using SQL, Numpy, and Numpy with C-accelerations

## Purpose
The primary purpose of these model is to rapidly prototype design ideas and compare their performance impacts. All the models assume a 'dataframe' approach to thinking about agents.

## SQL

TBD.

## Numpy

TBD.

## Numpy with C Accelerations

## How To Run

0. Install the required package:
    ```
    pip3 install sparklines
    ```

1. Build input files based on population size, node count, and EULA age specified in `settings.py`:
    ```
    make
    ```

2. Run the program:
    ```
    make run
    ```

### Intrahost

Note that the code is hardcoded to run an SEIR model. The settings.py controls base infectivity, birth rate, and some tuning parameters for how often births, deaths, migration, and RIA are calculated. 

Incubation duration, infectious duration are hardcoded in the code right now. Incubation is typically a constant value of 3. Infection is a uniform distribution draw.

### Node/Spatial Structure

We have the number of nodes specified in settings.py, and encoded in the population csvs. The nodes all have differerent populations, larger to smaller. Heterogeneity. All transmission is within node. 

### Migration

Migration is linear and one-directional. 1% of infected individuals are migrated "to the right" every 'migration_interval'. The idea here is have a simple and reliable (and easily visualizable and verifiable) way of doing spatial modeling.

### Seeding

We seed 100 infections in the largest node, and with our simplified migration model, it "bleeds to the right". It should function as a loop in fact with infection circulating back to the largest node from the smallest. If we encounter total fade-out (eradication), we re-seed w/ 100 infections in the largest node.

### Fertilty

Set Crude Birth Rate in settings.py. Births are calculated every 'fertility_interval' in settings.py. 

### Mortality

Non-Disease Mortality is calculated as an increasing function of age using a Gompertz-Makeham distribution applied to the EULA age-binned population (only right now). This obviously gets increasingly inaccurate as the simulation gets older. Deaths are calculated every 'mortality_interval' in settings.py. 

### Visualization

We use sparklines on each timestep to visualize the prevalence in each node. This provides a very cheap and simple way of getting real-time spatial visualzation.

