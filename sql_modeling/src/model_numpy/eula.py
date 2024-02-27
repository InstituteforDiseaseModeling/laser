import numpy as np
from collections import defaultdict
import settings
import ctypes
import pdb

update_ages_lib = ctypes.CDLL('./update_ages.so')
update_ages_lib.progress_natural_mortality_binned.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), # eula
    ctypes.c_size_t,  # num_nodes
    ctypes.c_size_t,  # num_age_bins
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # probs
    ctypes.c_size_t,  # timesteps_elapsed
]

eula = defaultdict(lambda: defaultdict(int))

# Define Gompertz-Makeham parameters
makeham_parameter = 0.01
gompertz_parameter = 0.05
age_bins = np.arange(15, 102)
probability_of_dying = 2.74e-6 * ( makeham_parameter + np.exp(gompertz_parameter * (age_bins - age_bins[0])) )

# call out to c function for this counting
def count_by_node_and_age( nodes, ages ):
    print( "Counting eulas by node and age; This is slow for now." )
    counts = defaultdict(lambda: defaultdict(int))
    for node_id, age in zip( nodes, ages ):
        age_bin = int(age)
        #age_bin = 44 # you can test out sticking everyone in a single bin
        counts[node_id][age_bin] += 1
    return counts

def init():
    global eula
    header_row = np.genfromtxt(settings.eula_file, delimiter=',', dtype=str, max_rows=1)

    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(settings.eula_file, delimiter=',', dtype=float, skip_header=1)

    #pdb.set_trace()
    for row in data:
        node = int(row[0])
        age = int(float(row[1]))  # Convert string to float and then to int
        total = int(row[2])
        if node not in eula:
            eula[node] = {}
        eula[node][age] = total

def progress_natural_mortality( timesteps ):
    def python():
        def get_simple_death_rate( age_bin_in_yrs ): # This obviously needs to be done once and then returned from a lookup table.
            # Calculate the probability of dying using the Gompertz-Makeham distribution
            # Gompertz-Makeham distribution: hazard = makeham_parameter + exp(gompertz_parameter * age)
            try:
                return probability_of_dying[ age_bin_in_yrs-15 ] # hack 15 is eula age for now
            except Exception as ex:
                pdb.set_trace()

        return_deaths = defaultdict(int)
        for node, age_bins_counts in eula.items():
            for age_bin, count in age_bins_counts.items():
                # Reduce count by 0.1%
                if eula[node][age_bin] > 0:
                    from scipy.stats import poisson, binom
                    prob = get_simple_death_rate( age_bin ) 
                    expected_deaths = sum(np.random.binomial(eula[node][age_bin], prob) for _ in range(timesteps))
                    if expected_deaths > 0:
                        #print( f"Killing off {expected_deaths} in node {node} and age_bin {age_bin} from existing population {eula[node][age_bin]} from prob {prob}." )
                        eula[node][age_bin] -= expected_deaths # round(count * (1-))
                        if eula[node][age_bin] == 0:
                            print( f"EULA bin for node {node} and age {age_bin} is now 0." )
                            #pdb.set_trace()
                    return_deaths[node] += expected_deaths 
        return return_deaths
    def c():
        pdb.set_trace()
        return update_ages_lib.progress_natural_mortality_binned(
            # TBD: SORT
            eula.values(), # sorted values as an array
            settings.num_nodes,
            len(eula.values()),  # size of eula array
            probability_of_dying,
            timesteps_elapsed )

def get_recovereds_by_node():   
    summary = {}
    for node in eula:
        summary[ node ] = sum( eula[node].values() )
    return summary
