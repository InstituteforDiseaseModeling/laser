# Proof-of-Concept Models Using SQL, Numpy, and Numpy with C-accelerations

## Purpose
The primary purpose of these models is to rapidly prototype design ideas and to measure and compare their performance impacts. All the models assume a 'dataframe' approach to thinking about agents.

## Approach

Each model consists of some number of agents -- we're trying to use no less than 100k and up to 10M for now. Each agent has a set of pre-defined attributes. The best way to think of this is as a big dataframe where the attributes are the column headers. The population can be fully represented in a csv file. The set of attributes we have in the code right now are (along with a sample row):

| ID | Node | Age | Infected | Immune | Incubation_Timer | Infection_Timer | Immune_Timer |
|----|------|-----|----------|--------|------------------|-----------------|--------------|
| 0  |  4   | 23.4|  false   |  true  |        0         |        0        |    1800      |

The user can attributes and code to operate on those attributes, but that's not demonstrated here yet.

It should be essentially trivial to write the entire model state to disk and reload that as one's initial population.

## Input Files
The SQL model can be bootstrap entirely from parameters in settings.py, but we are trying to standardize on a common input population csv file across all models, at least as an experiment. We imposes a startup cost but provides a simple startup story.

## SQL

The SQL model stores the entire model state in SQL table. We've been primarily been using SQLite, and in-memory db. We have a MySQL version we did to compare performance, but that is not be actively maintained. All model state updates are done as SQL statements. We are interested in how large we can make the population while still getting adequate performance and also exporting the usability of long-standing declarative modes of interaction. It's worth noting that the model state and any and every timestep is stored in the database (table) and available for query and inspection. If one uses an on-disk db instead of in-memory, the model state can be inspected externally with whatever csv/dataframe/SQL analysis tools one prefers.

## Numpy

The Numpy model represents all the model data as a set of numpy arrays. One can still think about the model the same way but one has to be able to code in numpy. This implementation is faster than the SQL version but we want to quantify how much faster, across a realistic range of model features and capabilities, and what are the end-user impacts on usability, model development, and debugging.

## Numpy with C Accelerations

Finally we have a version of the numpy model where most of the vectorized math is done in compiled C extensions. This is a big win on performance, but with a cost to end-user scrutibility. The installation process will include a c compilation step.

## How To Run

0. Make sure your pip is configured to check 
    ```
    https://packages.idmod.org/api/pypi/pypi-production/simple
    ```

1. The SQL model can be run by installing the latest dev package:

    ```
    pip3 install laser_sql_model
    ```

   The numpy model can be obtained by:
    ```
    pip3 install laser_numpy_model
    ```

   The C-Accelerated Numpy model is not yet available for pip installation. You should clone this repo, checkout the 'better_math' branch, and:
    ```
    pip3 install -r requirements.txt
    ```

2. For the numpy models, build input files based on population size, node count, and EULA age specified in the 'pop', 'num_nodes', and 'eula_age' params at the top of `settings.py`:
    ```
    make
    ```

3. Run the program:
    ```
    make run
    ```
    The simulation duration is controlled by the value of 'duration' in settings.py. It defaults to 20 years.

### Intrahost

Note that the code is hardcoded to run an SEIR model. The settings.py controls base base_infectivity, birth rate, and some tuning parameters for how often births, deaths, migration, and RIA are calculated. 

Incubation duration, infectious duration are hardcoded in the code right now. Incubation is typically a constant value of 3. Infection is a uniform distribution draw.

### Node/Spatial Structure

We have the number of nodes specified in settings.py, and encoded in the population csvs. The nodes all have differerent populations, larger to smaller. Heterogeneity. All transmission is within node. 

### Migration

Migration is linear and one-directional. 1% of infected individuals are migrated "to the right" every 'migration_interval'. The idea here is have a simple and reliable (and easily visualizable and verifiable) way of doing spatial modeling.

### Seeding

We seed 100 infections in the largest node, and with our simplified migration model, it "bleeds to the right". It should function as a loop in fact with infection circulating back to the largest node from the smallest. If we encounter total fade-out (eradication), we re-seed w/ 10 infections in a random node; this may fade out immediately or 'take' depending on epi factors in that and neighboring nodes at the time.

### Fertilty

Set Crude Birth Rate in settings.py. Births are calculated every 'fertility_interval' in settings.py. 

### Mortality

Non-Disease Mortality is calculated as an increasing function of age using a Gompertz-Makeham distribution applied to the EULA age-binned population (only right now). This obviously gets increasingly inaccurate as the simulation gets older. Deaths are calculated every 'mortality_interval' in settings.py. 

### Interventions

At this time you can distribute acquisition-blocking interventions to the adult population one time specified by 'campaign_day' in settings.py, with coverage 'campaign_coverage' (also settings.py) to a particular node ('campaign_node'). The values of campaign_coverage and campaign_node will also be used for RIA (at 9months). You can effectively disable the campaign and ria by setting campaign_day to a value greater than 'duration' or setting ria_interval to a value greater than 'duration'.

### Visualization

We use sparklines on each timestep to visualize the prevalence in each node. This provides a very cheap and simple way of getting real-time spatial visualzation.

The sims produce an output file called 'simulation_output.csv' with the values of Susceptibles, Infecteds, Recovereds, New Births, and New Deaths for each node and timestep. Visualizing that data is left as an exercise to the reader, though there are some pyplot-based utilities in the viz folder.
