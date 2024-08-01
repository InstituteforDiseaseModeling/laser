import numpy as np
import numba as nb

def add(model, count_births, istart, iend):
    # Randomly set ri_timer for coverage fraction of agents to a value between 8.5*30.5 and 9.5*30.5 days
    ri_timer_values = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16)  # 9mo-ish
    # Randomly set ri_timer for the second coverage fraction to a different range
    ri_timer_values2 = np.random.uniform(14.5 * 30.5, 15.5 * 30.5, count_births).astype(np.uint16)  # 15mo-ish

    mask = np.random.rand(count_births) < (model.nodes.ri_coverages[model.population.nodeid[istart:iend]])

    # Subdivision mask
    subdivision_rand = np.random.rand(mask.sum())

    # Create the three groups based on subdivision
    group_85 = subdivision_rand < 0.85
    group_14_25 = (subdivision_rand >= 0.85) & (subdivision_rand < 0.9925)

    # Directly apply the ri_timer values using the masks
    selected_indices = np.where(mask)[0]

    # Apply ri_timer_values to 85% of the selected agents
    model.population.ri_timer[istart:iend][selected_indices[group_85]] = ri_timer_values[selected_indices[group_85]]

    # Apply ri_timer_values2 to 14.25% of the selected agents
    model.population.ri_timer[istart:iend][selected_indices[group_14_25]] = ri_timer_values2[selected_indices[group_14_25]]


def add_with_ips(model, count_births, istart, iend):
    # Define the timer ranges
    ri_timer_values_9mo = np.random.uniform(8.5 * 30.5, 9.5 * 30.5, count_births).astype(np.uint16) # 9mo-ish
    ri_timer_values_15mo = np.random.uniform(14.5 * 30.5, 15.5 * 30.5, count_births).astype(np.uint16) # 15mo-ish

    # Get the accessibility values for the current range
    accessibility = model.population.accessibility[istart:iend]

    # Masks based on accessibility
    mask_0 = (accessibility == 0)
    mask_1 = (accessibility == 1)
    mask_2 = (accessibility == 2) # for validation
    if np.count_nonzero( mask_1 ) == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 1 (medium)." )
    if np.count_nonzero( mask_2 ) == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 2 (medium)." )

    # mask_2 is unnecessary since we don't apply any timer for it

    # Apply the 9-month-ish timer to accessibility 0
    model.population.ri_timer[istart:iend][mask_0] = ri_timer_values_9mo[mask_0]

    # Apply the 15-month-ish timer to accessibility 1
    model.population.ri_timer[istart:iend][mask_1] = ri_timer_values_15mo[mask_1]

    # No need to apply anything to accessibility 2, as it should remain unmodified

    return

# Define the function to decrement ri_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.uint16[:], nb.int32[:], nb.int64 ), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility, age_at_vax, dob, tick):
    for i in nb.prange(count):
        if ri_timer[i] > 0:
            ri_timer[i] -= 1
            # TBD: It's perfectly possible that the individual got infected (or recovered) while this timer
            # was counting down and we might want to abort the timer.
            if ri_timer[i] == 0:
                susceptibility[i] = 0
                #age_at_vax[i] = tick-dob[i] # optional for reporting

