import numpy as np
import numba as nb

def add( model, count_births, istart, iend ):
    # Randomly set ri_timer for coverage fraction of agents to a value between 8.5*30.5 and 9.5*30.5 days
    # change these numbers or parameterize as needed
    ri_timer_values = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16) # 9mo-ish
    # Randomly set ri_timer for the second coverage fraction to a different range
    ri_timer_values2 = np.random.uniform(14.5 * 30.5, 15.5 * 30.5, count_births).astype(np.uint16) # 15mo-ish

    mask = np.random.rand(count_births) < (model.nodes.ri_coverages[model.population.nodeid[istart:iend]])

    # Create a mask to select coverage fraction of agents
    # Do coverage by node, not same for every node
    # I don't think agents have node ids yet?
    # Subdivision mask
    subdivision_rand = np.random.rand(mask.sum())

    # Create the three groups based on subdivision
    group_85 = subdivision_rand < 0.85
    group_14_25 = (subdivision_rand >= 0.85) & (subdivision_rand < 0.9925)
    group_0_75 = subdivision_rand >= 0.9925

    #import pdb
    #pdb.set_trace()
    # Apply ri_timer_values to 85% of the selected agents
    model.population.ri_timer[istart:iend][mask][group_85] = ri_timer_values[mask][group_85]

    # Apply ri_timer_values2 to 14.25% of the selected agents
    model.population.ri_timer[istart:iend][mask][group_14_25] = ri_timer_values2[mask][group_14_25]

# Define the function to decrement ri_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility):
    for i in nb.prange(count):
        if ri_timer[i] > 0:
            ri_timer[i] -= 1
            # TBD: It's perfectly possible that the individual got infected (or recovered) while this timer
            # was counting down and we might want to abort the timer.
            if ri_timer[i] == 0:
                susceptibility[i] = 0

