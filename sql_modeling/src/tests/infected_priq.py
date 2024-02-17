import numpy as np
import heapq
from collections import defaultdict
import pdb
 
# Allocate an array of 1 million boolean values
SIZE = int(1e7)
array = np.zeros(SIZE, dtype=bool)
 
# Define the percentage of elements to randomly select at each timestep
percentage = 1e-3
 
# Define the number of timesteps
timesteps = 730
 
# Create a priority queue to store (time, index) pairs
priority_queue = []

def select_new_infecteds( array ):
    # Randomly select 1% of elements and set them to True
    random_indices = np.random.choice(SIZE, int(percentage * SIZE), replace=False)
    #array[random_indices[~np.where(array)[0]]] = True
    array[random_indices[~array[random_indices]]] = True
    return random_indices

def test_vectors():
    # Iterate over each timestep
    infection_timers = np.zeros_like(array, dtype=int)
    for timestep in range(timesteps):
        # infect a fraction of uninfecteds
        random_indices = select_new_infecteds( array )
        uninfected_indices = random_indices[infection_timers[random_indices]==0]
        # initialize their timers
        infection_timers[uninfected_indices] = np.random.randint(5, 16)
       
        # Decrement infection timers
        infection_timers[np.where(array)[0]] -= 1
    
        # Flip bools to 0 where infection timers are 0
        array[infection_timers == 0] = False 
        #print( f"T={timestep}, infected={np.count_nonzero(array)}." )

def test_priq():
    # Iterate over each timestep
    for timestep in range(timesteps):
        random_indices = select_new_infecteds( array )

        # For each selected index, generate a random timer value between 5 and 15
        for index in random_indices:
            timer_value = np.random.randint(5, 16)
            # Calculate the time when the timer will expire and add it to the priority queue
            heapq.heappush(priority_queue, (timestep + timer_value, index))
        
        # Check the priority queue to identify indices where the timer has expired
        while priority_queue and priority_queue[0][0] == timestep:
            _, expired_index = heapq.heappop(priority_queue)
            # Set the corresponding boolean value in the array to False
            #print( f"{expired_index} just cleared." )
            array[expired_index] = False
        #print( f"T={timestep}, infected={np.count_nonzero(array)}." )
     
def test_map():
    def empty_array():
        return np.array([]).astype( int )

    #infecteds = defaultdict(empty_array)
    infecteds = defaultdict(list)
     
    # Iterate over each timestep
    for timestep in range(timesteps):
        random_indices = select_new_infecteds( array )

        # Calculate timer values for each random index
        timer_values = np.random.randint(timestep + 5, timestep + 16, size=len(random_indices))

        # Calculate expiration times for each random index
        #expiration_times = timestep + timer_values
        expiration_times = timer_values

        # Find unique expiration times
        #unique_expiration_times = np.unique(timer_values)
        unique_expiration_times = [timestep + x for x in range( 5, 16 ) ]

        # Assign or extend the infecteds map with the corresponding indices for each expiration time
        for expiration_time, index in zip(expiration_times, random_indices):
            #infecteds[expiration_time] = np.concatenate((infecteds[expiration_time], [index]))
            infecteds[expiration_time].append( index )

        # Check the priority queue to identify indices where the timer has expired
        if timestep in infecteds:
            recovereds = np.array(infecteds[ timestep ])
            array[recovereds] = False
            del infecteds[ timestep ]
        #print( f"T={timestep}, infected={np.count_nonzero(array)}." )
     
from timeit import timeit

runtime = timeit( test_map, number=1 )
print( f"Map Execution time = {runtime}." )

array = np.zeros(SIZE, dtype=bool)
runtime = timeit( test_vectors, number=1 )
print( f"Vectors Execution time = {runtime}." )

#runtime = timeit( test_priq, number=1 )
#print( f"Priq Execution time = {runtime}." )

