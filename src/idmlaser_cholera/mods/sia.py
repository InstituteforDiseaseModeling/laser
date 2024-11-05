import numba as nb
import numpy as np
import json
import pdb

cdf_fit = None

# ## Interventions : Serosurveys and SIAs
# 
# Let's try an SIA 10 days into the simulation.
# 
# _Consider sorting the SIAs by tick in order to be able to just check the first N campaigns in the list._
# 

# In[24]:


from collections import namedtuple

SEROSURVEY = namedtuple("SEROSURVEY", ["tick", "nodes", "age_days_min", "age_days_max"])
SIA = namedtuple("SIA", ["tick", "nodes", "coverage", "age_days_min", "age_days_max"])
EDUCATION = namedtuple("EDUCATION", ["tick", "nodes"])

NINE_MONTHS = 274   # 9 months in days (30 day months + 9/12 of the 5 addition days in a year)
SIX_YEARS = 6 * 365 # 6 years in days
HUNDRED_YEARS = 100 * 365

todo = [
    #SEROSURVEY(9, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 9, nodes 0-5, [2-6) years old
    #SIA(10, [1, 3, 5], 0.80, NINE_MONTHS, SIX_YEARS), # Tick 10, nodes 1, 3, and 5, 80% coverage, [2-6) years old
    SIA(365, [0, 10, 20, 30, 40], 0.99, 0, HUNDRED_YEARS), # Tick 10, nodes 1, 3, and 5, 80% coverage, [2-6) years old
    #SEROSURVEY(11, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 11, nodes 0-5, [2-6) years old
    #EDUCATION(30, [0, 1, 2, 3, 4, 5]),  # Tick 30, nodes 0-5
    ]

def interpolate_duration(random_value, cdf_fit, time_fit):
    return np.interp(random_value, cdf_fit, time_fit)

def init( model, manifest ):
    model.nodes.add_vector_property("seronegativity", model.params.ticks, dtype=np.uint32)
    # Load the fitted parameters
    #with open("immune_decay_params.json", "r") as f:
    with open(manifest.immune_decay_params, "r") as f:
        params = json.load(f)

    # Extract the parameters
    phi = params["phi"]
    omega = params["omega"]
    tvaccination = params["tvaccination"]

    # Define the immune decay function
    def immune_decay(t, phi, omega, tvaccination):
        return phi * (1 - omega) ** (t - tvaccination)

    # Generate time array for CDF calculation
    time_fit = np.linspace(0, 5475, 1000)
    predicted_fit = immune_decay(time_fit, phi, omega, tvaccination)

    # Calculate the CDF from the predicted fit values
    global cdf_fit
    cdf_fit = np.cumsum(predicted_fit) / np.sum(predicted_fit)  # Normalize to create a CDF


@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.uint32[:]), parallel=True)
def _do_serosurvey(count, targets, nodeids, susceptibilities, seronegativity):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            if susceptibilities[i] > 0:
                seronegativity[nodeids[i]] += 1

    return

def invoke_serosurvey(campaign, model, tick):
    print(f"Running serosurvey {campaign=} at tick {tick}")
    targets = np.zeros(model.nodes.count, dtype=np.uint16)
    targets[campaign.nodes] = 1
    _do_serosurvey(
        model.population.count,
        targets,
        model.population.nodeid,
        model.population.susceptibility,
        model.nodes.seronegativity[:, tick],
        )

    return

@nb.njit((
    nb.uint32, # count
    nb.uint16[:], # targets
    nb.uint16[:], # nodeids
    nb.uint8[:], # susceptibilities
    nb.uint16[:], # sus_timers
    nb.float32, # coverage
    nb.int32[:], # dobs
    nb.int32, # age_min
    nb.int32, # age_max
    nb.float64[:], # cdf_fit
    nb.int32), # tick
parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, sus_timers, coverage, dobs, age_min, age_max, cdf_fit, tick):
    time_fit = np.linspace(0, 5475, 1000)
    def linear_interp(x, xp, fp):
        #if xp is None:
        #    raise ValueError( "cdf_fit {xp} is None" )
        #if fp is None:
        #    raise ValueError( "time_fit {fp} is None" )
        for i in range(len(xp) - 1):
            if xp[i] <= x < xp[i + 1]:
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])
        return fp[-1]  # Return last value if x is outside xp range

    #global cdf_fit
    #if cdf_fit is None:
    #    raise ValueError( "cdf_fit is None" )

    for i in nb.prange(count):
        if targets[nodeids[i]]:
            age = tick - dobs[i]
            if (age_min <= age) and (age < age_max):
                if np.random.random_sample() < coverage:
                    susceptibilities[i] = 0
                    random_value = np.random.uniform(0, 1)
                    duration = linear_interp(random_value, cdf_fit, time_fit)
                    sus_timers[i] = 90 + np.float32(duration) # box-decay in EMOD jargon
                    #print( f"[SIA] Setting susceptibility timer for {i} in node {nodeids[i]} to {sus_timers[i]}." )

    return

def invoke_sia(campaign, model, tick):
    print(f"Running SIA {campaign=} at tick {tick}")
    targets = np.zeros(model.nodes.count, dtype=np.uint16)
    # Use a boolean mask to filter the array
    targets[campaign.nodes] = 1
    global cdf_fit
    if cdf_fit is None:
        raise ValueError( "cdf_fit is None" )
    _do_sia(
        model.population.count,
        targets,
        model.population.nodeid,
        model.population.susceptibility,
        model.population.susceptibility_timer,
        np.float32(campaign.coverage),
        model.population.dob,
        np.int32(campaign.age_days_min),
        np.int32(campaign.age_days_max),
        cdf_fit,
        np.int32(tick),
        )

    return

iv_map = {
    SEROSURVEY: invoke_serosurvey,
    SIA: invoke_sia,
    EDUCATION: lambda campaign, model, tick: print(f"Running education {campaign=} at tick {tick}"),
}

def step(model, tick):
    while len(todo) > 0 and todo[0][0] == tick:
        campaign = todo.pop(0)
        campaign_nodes_np = np.array(campaign.nodes)
        campaign.nodes.clear()
        campaign.nodes.extend( campaign_nodes_np[campaign_nodes_np < model.nodes.count].tolist() ) # maybe not permanent code
        iv_map[type(campaign)](campaign, model, tick)

