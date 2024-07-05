# Welcome to IDMLaser

## Installation
```
pip3 install idmlaser
```

## QuickStart Setup

```
python3 -m idmlaser.utils.build_template_workspace
```

This will prompt you to specify a path to a sandbox directory to create a new workspace. 
```
Enter the sandbox directory path (default: /var/tmp/sandbox):
```
It will also prompt you if the sandbox directory already exists because it will wipe it clean if you consent. E.g.,
```
The directory '/var/tmp/sbox' already exists. Do you want to overwrite it? (yes/no):
```
And it will ask if you want to run the pre-configured England & Wales spatial scenario or a "Critical Community Size" synthetic scenario. You will see some console output as files are copied and/or downloaded and some initialization is done. Either of these should just work. The CCS is the simplest and fastest and recommended for first time users.

You can then change directory to the sandbox or workspace (use ```cd``` or ```pushd```) and run the model.

### Run Model

At the most basic level, to run the model, you do:

```
python3 -m idmlaser.measles
```

You should see some console output such as "T=<N>" for all the timesteps up to 7300 (20 years) as well as sparkline output for each timestep.

*Output*
The simulation should produce a ```simulation_output.csv``` report file. It consists of the following columns:
```
Timestep,Node,Susceptible,Infected,New_Infections,Recovered,Births
```

You can use any solution of your choice to plot the Infected, Susceptible, Recovered or New_Infections channels by Node over Time. If you're running the CCS demo, there should only be 1 node (node id=0). If you're running the E&W demo, it's most instructive at first to plot output for node 507 (London) or Birmingham (99).

## Examples

### Example 1: England & Wales

The E&W example is the classic "Measles in England and Wales during the post-war Period" dataset. It consists of 954 locations with location-specific birth rates. The model input files are downloaded during setup and do not not need to be regenerated. We put everyone over age 5 into an initial EULA bucket and model only the under 5s. The input modeled population (agents) file has size 495644. 

### Example 2: CCS (1 node)

The CCS 1 node example is the simplest example since it's not spatial. The total initial population is 2.4 million, again with the EULA age set at 5. This results in 118,722 agents initially being modeled. The duration is also 20 years. Birth rate is set at 17.5 (CBR).

### Example 3: CCS (100 nodes)

This example is achieved by starting with CCS 1 node and editing the demographics_settings.py file to change the total population to something like 1e7 and set the num_nodes to 100. Then type ```make``` to regenerate the model input files.

## Workflow

- Create Input Files

Modify demographics_settings.py. Set the pop(ulation) and num_nodes, and maybe eula_age.

```
make
```

- Edit Settings

See below section on parameters in settings.py for what you can change here.

- Run Model
```
python3 -m idmlaser.measles
```


## Input Files

The main disease model will look for at least two files:
- settings.py
- demographics_settings.py

These are both simple files with Python-compatible key-value pairs. Let's start with demograhics_settings.py. There is an example in examples/demographics_settings.py.
- pop_file: filename of compressed csv with all the agents to be modeled. Columns are attributes. Rows are agents.
- eula_pop_fits: filename of npy file which is the slope and intercept of the eula population over time by node.
- cbr_file: filename of csv file with the crude birth rates of each node by year.

- nodes: an array (list) with the all the node ids. Could be as simple as '[0]'
- num_nodes: The total number of nodes. (Could be inferred from pop_file or even nodes.)
- eula_age: The EULA age threshold. EULA=Epidemiologically Uninteresting Light Agents. There is nobody older than this in the original modeled population. This is used as an input for the model pre-proc step.


The settings.py file consists of:
- duration: Simulation duration in days.
- base_infectivity
- seasonal_multiplier: Scalar to apply to annual seasonality multiplier curve
- infectivity_multiplier: Array of multipliers representing seasonality.

*Reporting*
- report_filename: Defaults to ="simulation_output.csv"
- report_start: When to start reporting

*Burnin*
- burnin_delay: Delay from start to wait before injecting cases
- import_cases: not used
- dont_import_after: Time after which to stop importing


*Runtime Demographics*
- cbr: Crude Birth Rate. Set this to -1 to use cbr by year and node via cbr_file.
- mortality_interval: Timesteps between applying natural mortality (can remove)
- fertility_interval: Timesteps between adding babies (can probably remove)

*Migration*
- attraction_probs_file: csv file with probabilities of agent traveling from node A to node B.
- migration_fraction: Fraction of infected people to migrate to another node each migration.
- migration_interval: Days between migrations


*Interventions (experimental)*
- campaign_day: Day to launch test SIA campaign
- campaign_coverage: Coverage for test SIA campaign
- campaign_node: Node for test SIA campaign
- ria_interval: Days between RIA distributions

## Model Behavior

### Default Model Behavior

The model behavior is essentially defined by the properties and step functions. The properties that are currently hardcoded in this package are:

```
['id', 'node', 'age', 'infected', 'infection_timer', 'incubation_timer', 'immunity', 'immunity_timer', 'expected_lifespan' ]
```

These are ultimately controlled by the code in the idmlaser.utils.create_pop_as_csv tool and seen as the columns of the modeled_pop.csv.gz files. For example, in this particular implmentation, each agent has an "age" which turns out to be in units of years. There is code in "update_ages.cpp" which assumes the existence of an age column. Most obviously the "update_ages" function itself, which ages people by 1 day each day, but other step functions in update_ages.cpp may check the agent's age. Each agent has an infected boolean flag and an immunity flag. Those each have countdown timers. The immunity_timer can be set to -1 for permanent immunity. A positive countdown timer gets counted down each timestep by 1 and the flag gets set to False/0 when the timer reaches 0. That's not a fundamental decision of this design, just what is implemented in the code right now. 

### New Model Behavior
One can completely redesign the behavior of the model.  To modify the model behavior you can:
- Add/remove properties.
- Add/remove/modify code in update_ages.cpp (and recompile).
- Add/remove/modify glue code in sir_numpy_c.

If, say, you wanted to model agents ages but use a fixed date-of-birth (dob) and caclulate age on-the-fly by comparing to "now", you would need to create the model with a dob column (and assign values during initialization and at birth) and also modify the step functions.

I have made no attempt up to this point to create an infrastructure that is 100% agnostic or dynamic on model attributes (columns). Making the code more generic and abstracted will also make it a bit more complex. "There are no solutions, only tradeoffs".

Let's consider all the places that the code currently "knows" that there is an "age" column, i.e., where it's hardcoded and would need to be changed if age was done differently:

Init:
- [Model dataframe initialization](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/utils/create_pop_as_csv.py#L27)
- [Loading model dataframe into np array](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy.py#L49)
- [Adding 'expansion slots'](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L200)
 
Stepwise:
- Age everyone (already born): [py](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L300) and [C](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/update_ages.cpp#L55)
- Make newborns: [py](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L330) and [C](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/update_ages.cpp#L529)
- RIA: [py](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L524) and [C](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/update_ages.cpp#L473)
- SIA: [py](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L537) and [C](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/update_ages.cpp#L439)
- Collect Report: [py](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/sir_numpy_c.py#L264) and [C](https://github.com/InstituteforDiseaseModeling/laser/blob/jb_modulify/jb/src/idmlaser/update_ages.cpp#L295)

Each of the "stepwise" functions also have a argtype declaration at the top of sir_numpy_c.py which is aware of the age column.

I shall not repeat that for each of the other attributes/properties (e.g., infected, incubation_timer).
## 
