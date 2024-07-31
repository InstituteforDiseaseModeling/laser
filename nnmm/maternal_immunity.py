import numpy as np
import numba as nb

def add( model, istart, iend ):
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(0.5*365) # 6 months

# Define the function to decrement susceptibility_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_sus_timer(count, susceptibility_timer, susceptibility):
    for i in nb.prange(count):
        if susceptibility_timer[i] > 0:
            susceptibility_timer[i] -= 1
            if susceptibility_timer[i] == 0:
                susceptibility[i] = 1

