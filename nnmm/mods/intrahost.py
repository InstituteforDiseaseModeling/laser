import numba as nb
import numpy as np

# ## Incubation and Infection
# 
# We will add incubation timer (`etimer` for "exposure timer") and an infection (or infectious) timer (`itimer`) properties to the model population. A `uint8` counting down from as much as 255 days will be more than enough.
# 
# We wrap a Numba compiled function using all available cores in the infection update function, extracting the `count` and `itimer` values the JITted function needs.
# 
# Similarly, we wrap a Numba compiled function using all available cores in the exposure update function, extracting the values the JITted function needs from the `model` object.

# In[13]:


@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
def _exposure_update(count, etimers, itimers, inf_mean, inf_std):
    for i in nb.prange(count):
        if etimers[i] > 0:
            etimers[i] -= 1
            if etimers[i] == 0:
                itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))    # must be at least 1 day

    return

def do_exposure_update(model, tick):

    _exposure_update(nb.uint32(model.population.count), model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std)

    return


@nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint8[:]), parallel=True)
def _infection_update(count, itimer, sus_timer, susceptibility):
    for i in nb.prange(count):
        if itimer[i] > 0:
            itimer[i] -= 1
            if itimer[i] <= 0:
                susceptibility = 0
                sus_timer[i] = 1*365

    return

def do_infection_update(model, tick):

    _infection_update(nb.uint32(model.population.count), model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility)

    return


