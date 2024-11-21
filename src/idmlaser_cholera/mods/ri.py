import numpy as np
import numba as nb
import ctypes

# ## Routine Immunization (RI) 
# Do MCV1 for most kids at around 9mo. Do MCV2 for most of the kids who didn't take around 15mo. There are a bunch of ways of doing this.
# What we're doing here is using an IP -- to borrow EMOD terminology -- for Accessibility, with values 0 (Easy), 1 (Medium), 2 (Hard). See elsewhere for that.
# The MCV1 Timer is a uniform draw from 8.5 months to 9.5 months. MCV2 Timer is a uniform draw from 14.5 months to 15.5 months.
# ### Coverages
# All MCV is subject to coverage levels which vary by node.

# ## RI
# ## Accessibility "IP" Groups
# - 85% of in-coverage kids get MCV1
# - 14% get MCV2
# - 1% get nothing
# Vary as needed.

# In[16]:


# ## RI 
# ### Add based on accessibility group for newborns

# Timers get counted down each timestep and when they reach 0, susceptibility is set to 0.

use_nb = True
lib = None

def init( model ):
    # Add new property "ri_coverages", just randomly for demonstration purposes
    # Replace with values from data
    model.nodes.add_scalar_property("ri_coverages", dtype=np.float32)
    model.nodes.ri_coverages = np.random.rand(len(model.nn_nodes))

    try:
        global lib
        lib = ctypes.CDLL('./libri.so')

        # Define the argument types for the C function
        lib.update_susceptibility_based_on_ri_timer.argtypes = [
            ctypes.c_uint32,                  # count
            np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # ri_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            #np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # age_at_vax
            #np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),   # dob
            #ctypes.c_int64                    # tick
        ]
        global use_nb
        use_nb = False
    except Exception as ex:
        print( "Failed to load libri.so. Will use numba." )


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
    """
    if np.count_nonzero( mask_1 ) == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 1 (medium)." )
    if np.count_nonzero( mask_2 ) == 0:
        raise ValueError( "Didn't find anyone with accessibility set to 2 (medium)." )
    """

    # mask_2 is unnecessary since we don't apply any timer for it

    # Apply the 9-month-ish timer to accessibility 0
    model.population.ri_timer[istart:iend][mask_0] = ri_timer_values_9mo[mask_0]

    # Apply the 15-month-ish timer to accessibility 1
    model.population.ri_timer[istart:iend][mask_1] = ri_timer_values_15mo[mask_1]

    # No need to apply anything to accessibility 2, as it should remain unmodified

    return


# Define the function to decrement ri_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.int32[:], nb.int64 ), parallel=True)
def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility, dob, tick):
    for i in nb.prange(count):
        if ri_timer[i] > 0:
            ri_timer[i] -= 1
            # TBD: It's perfectly possible that the individual got infected (or recovered) while this timer
            # was counting down and we might want to abort the timer.
            if ri_timer[i] == 0:
                susceptibility[i] = 1
                #age_at_vax[i] = tick-dob[i] # optional for reporting



#def _update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility, dob, tick):
    #_update_susceptibility_based_on_ri_timer(count, ri_timer, susceptibility, dob, tick)

def do_ri(model, tick):
    if use_nb:
        _update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility, model.population.dob, tick)
    else:
        lib.update_susceptibility_based_on_ri_timer(model.population.count, model.population.ri_timer, model.population.susceptibility)
    return

