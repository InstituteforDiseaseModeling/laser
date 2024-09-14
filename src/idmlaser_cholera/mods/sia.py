import numba as nb
import numpy as np


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

todo = [
    SEROSURVEY(9, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 9, nodes 0-5, [2-6) years old
    SIA(10, [1, 3, 5], 0.80, NINE_MONTHS, SIX_YEARS), # Tick 10, nodes 1, 3, and 5, 80% coverage, [2-6) years old
    SEROSURVEY(11, [0, 1, 2, 3, 4, 5], NINE_MONTHS, SIX_YEARS),  # Tick 11, nodes 0-5, [2-6) years old
    EDUCATION(30, [0, 1, 2, 3, 4, 5]),  # Tick 30, nodes 0-5
    ]

def init( model ):
    model.nodes.add_vector_property("seronegativity", model.params.ticks, dtype=np.uint32)

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

@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.int32[:], nb.int32, nb.int32, nb.int32), parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, coverage, dobs, age_min, age_max, tick):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            age = tick - dobs[i]
            if (age_min <= age) and (age < age_max):
                if susceptibilities[i] > 0:
                    if np.random.random_sample() < coverage:
                        susceptibilities[i] = 0

    return

def invoke_sia(campaign, model, tick):
    print(f"Running SIA {campaign=} at tick {tick}")
    targets = np.zeros(model.nodes.count, dtype=np.uint16)
    targets[campaign.nodes] = 1
    _do_sia(
        model.population.count,
        targets,
        model.population.nodeid,
        model.population.susceptibility,
        np.float32(campaign.coverage),
        model.population.dob,
        np.int32(campaign.age_days_min),
        np.int32(campaign.age_days_max),
        np.int32(tick),
        )

    return

iv_map = {
    SEROSURVEY: invoke_serosurvey,
    SIA: invoke_sia,
    EDUCATION: lambda campaign, model, tick: print(f"Running education {campaign=} at tick {tick}"),
}

def do_interventions(model, tick):
    while len(todo) > 0 and todo[0][0] == tick:
        campaign = todo.pop(0)
        iv_map[type(campaign)](campaign, model, tick)

