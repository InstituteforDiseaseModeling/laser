import numpy as np
from collections import defaultdict
import settings
import demographics_settings

eula_dict = defaultdict(lambda: defaultdict(int))

# Define Gompertz-Makeham parameters
makeham_parameter = 0.01
gompertz_parameter = 0.05
age_bins = np.arange(demographics_settings.eula_age, 102)
probability_of_dying = 2.74e-6 * ( makeham_parameter + np.exp(gompertz_parameter * (age_bins - age_bins[0])) )
#print( f"probability_of_dying = {probability_of_dying}" )
fits = np.load(demographics_settings.eula_pop_fits, allow_pickle=True).item()
def calculate_y(x, m, b):
    return int(m * x + b)

timestep_abs = 0
next_eula_pops = np.zeros( demographics_settings.num_nodes ).astype( np.uint32 )

# call out to c function for this counting
def count_by_node_and_age( nodes, ages ):
    print( "Counting eulas by node and age; This is slow for now." )
    counts = defaultdict(lambda: defaultdict(int))
    for node_id, age in zip( nodes, ages ):
        age_bin = int(age)
        counts[node_id][age_bin] += 1
    return counts

def init():
    global eula_dict, next_eula_pops 

    for node in range(settings.num_nodes):
        m, b = fits[node]
        pop = calculate_y(0, m, b)
        # print( f"Setting pop for node {node} to {pop}." )
        next_eula_pops[ node ] = pop
    

def progress_natural_mortality( timesteps ):
    global timestep_abs 
    timestep_abs += timesteps

    def from_lut():
        # Calculate y values using the fit parameters and x values
        global next_eula_pops 
        for node in range(settings.num_nodes):
            m, b = fits[node]
            pop = calculate_y(timestep_abs, m, b)
            #print( f"pop={pop},m={m},b={b},node={node},time={timestep_abs}" )
            #eula_dict[node][44] = pop
            next_eula_pops[ node ] = pop
    from_lut()


def get_recovereds_by_node_np():
    return next_eula_pops

