import numpy as np

def set_accessibility(model, istart, iend ):
    # Get the coverage probabilities for the relevant agents
    coverages = model.nodes.ri_coverages[model.population.nodeid[istart:iend]]
    
    # Generate random numbers to decide accessibility based on coverages
    random_values = np.random.rand(coverages.size)

    # Calculate attenuated probabilities
    prob_0 = 0.85 * coverages
    prob_1 = 0.14 * coverages
    prob_2 = 0.01 * coverages

    # Normalize probabilities to ensure they sum to 1 after attenuation
    total_prob = prob_0 + prob_1 + prob_2
    prob_0 /= total_prob
    prob_1 /= total_prob
    prob_2 /= total_prob

    # Initialize accessibility array with zeros
    accessibility = np.zeros_like(random_values, dtype=np.uint8)

    # Set values based on attenuated probabilities
    accessibility[random_values < prob_0] = 0
    accessibility[(random_values >= prob_0) & (random_values < prob_0 + prob_1)] = 1
    accessibility[random_values >= prob_0 + prob_1] = 2

    # Assign to the population's accessibility attribute
    model.population.accessibility[istart:iend] = accessibility

