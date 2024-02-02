import numpy as np
from collections import defaultdict

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

def init( nodes, ages, age_threshold_yrs ):
    global eula
    eula = count_by_node_and_age( nodes[ages>=age_threshold_yrs], ages[ages>=age_threshold_yrs] )

def progress_natural_mortality():
    def get_simple_death_rate( age_bin_in_yrs ):
        # This obviously needs to be done once and then returned from a lookup table.

        # Calculate the probability of dying using the Gompertz-Makeham distribution
        # Gompertz-Makeham distribution: hazard = makeham_parameter + exp(gompertz_parameter * age)
        try:
            return probability_of_dying[ age_bin_in_yrs-15 ] # hack 15 is eula age for now
        except Exception as ex:
            pdb.set_trace()

    # first highly simplistic just to get the plumbing working
    for node, age_bins_counts in eula.items():
        for age_bin, count in age_bins_counts.items():
            # Reduce count by 0.1%
            if eula[node][age_bin] > 0:
                from scipy.stats import poisson, binom
                prob = get_simple_death_rate( age_bin )
                expected_deaths = np.random.binomial(eula[node][age_bin], prob)
                if expected_deaths > 0:
                    #print( f"Killing off {expected_deaths} in node {node} and age_bin {age_bin} from existing population {eula[node][age_bin]} from prob {prob}." )
                    eula[node][age_bin] -= expected_deaths # round(count * (1-))
                    if eula[node][age_bin] == 0:
                        print( f"EULA bin for node {node} and age {age_bin} is now 0." )
                        #pdb.set_trace()
