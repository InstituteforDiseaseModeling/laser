import numpy as np
import numba as nb

# initialize susceptibility based on age
@nb.njit((nb.uint32, nb.int32[:], nb.uint8[:], nb.uint16[:]), parallel=True)
def initialize_susceptibility(count, dob, susceptibility, susceptibility_timer):

    for i in nb.prange(count):
        susceptibility[i] = 0
        # Initialize everyone's immunity timer uniformly from -5 to +5 years,
        # where negative =0 (to get half population initially susceptible)
        # TBD: Replace this with correct formula
        timer = np.random.randint(-5*365, 5*365)
        if timer <= 0:
            timer = 0
            susceptibility[i] = 1
        else:
            susceptibility_timer[i] = timer

    return

def init( model ):
    return initialize_susceptibility( model.population.count, model.population.dob, model.population.susceptibility, model.population.susceptibility_timer )
