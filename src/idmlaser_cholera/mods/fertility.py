import numpy as np
import pdb
from . import access_groups as ag
from . import ri
from . import maternal_immunity as mi

from idmlaser_cholera.kmcurve import pdsod

# ## Vital Dynamics: Births
# 
# Let's implement births over time. We will use the CBR in `model.params` and draw for the number of births this year based on the most recent population. Then, we will distribute those births as evenly as possible for integral values over the days of the year.
# 
# Note that we add in the date of birth and date of non-disease death after we add those properties below.
# 
# Note that we add in initializing the susceptibility after we add that property below.


def step(model, tick):

    doy = tick % 365 + 1    # day of year 1...365
    year = tick // 365

    if doy == 1:
        if model.nodes.cbrs is not None:
            # cbr by node
            model.nodes.births[:, year] = np.random.poisson(model.nodes.population[:, tick] * model.nodes.cbrs / 1000)
        else:
            model.nodes.births[:, year] = np.random.poisson(model.nodes.population[:, tick] * model.params.cbr / 1000)
        #print( f"Births for year {year} = {model.nodes.births[:, year]}" )

    annual_births = model.nodes.births[:, year]
    todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
    count_births = todays_births.sum()
    istart, iend = model.population.add(count_births)   # add count_births agents to the population, get the indices of the new agents

    # enable this after loading the aliased distribution and dod and dob properties (see cells below)
    model.population.dod[istart:iend] = pdsod(model.population.dob[istart:iend], max_year=100)   # make use of the fact that dob[istart:iend] is currently 0
    model.population.dob[istart:iend] = tick    # now update dob to reflect being born today

    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 1

    # Randomly set ri_timer for coverage fraction of agents to a value between 8.5*30.5 and 9.5*30.5 days
    # change these numbers or parameterize as needed
    ri_timer_values = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16)

    # Create a mask to select coverage fraction of agents
    # Do coverage by node, not same for every node
    try:
        mask = np.random.rand(count_births) < (model.nodes.ri_coverages[model.population.nodeid[istart:iend]])
        # Set ri_timer values for the selected agents
        model.population.ri_timer[istart:iend][mask] = ri_timer_values[mask]
    except Exception as ex:
        print( "Exception setting ri_timers for newborns." )
        print( str( ex ) )
        pdb.set_trace()


    index = istart
    nodeids = model.population.nodeid   # grab this once for efficiency
    dods = model.population.dod # grab this once for efficiency
    max_tick = model.params.ticks
    for nodeid, births in enumerate(todays_births):
        nodeids[index:index+births] = nodeid
        for agent in range(index, index+births):
            # If the agent will die before the end of the simulation, add it to the queue
            if dods[agent] < max_tick:
                model.nddq.push(agent)
        index += births
    model.nodes.population[:,tick+1] += todays_births

    ag.set_accessibility(model, istart, iend )
    ri.add_with_ips( model, count_births, istart, iend )
    mi.init( model, istart, iend )
    return

