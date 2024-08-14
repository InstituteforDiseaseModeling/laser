import numba as nb
import numpy as np

# ## SIAs
# 
# Let's try an SIA 10 days into the simulation.
# 
# _Consider sorting the SIAs by tick in order to be able to just check the first N campaigns in the list._
# 

sias = [(10, [1, 3, 5], 0.80)]  # Tick 10, nodes 1, 3, and 5, 80% coverage.

@nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32), parallel=True)
def _do_sia(count, targets, nodeids, susceptibilities, coverage):
    for i in nb.prange(count):
        if targets[nodeids[i]]:
            if susceptibilities[i] > 0:
                if np.random.random_sample() < coverage:
                    susceptibilities[i] = 0
    return

def do_sias(model, tick):
    while len(sias) > 0 and sias[0][0] == tick:
        campaign = sias.pop(0)
        print(f"Running SIA {campaign=} at tick {tick}")
        (day, nodes, coverage) = campaign
        targets = np.zeros(model.nodes.count, dtype=np.uint8)
        targets[nodes] = 1
        _do_sia(model.population.count, targets, model.population.nodeid, model.population.susceptibility, coverage)


