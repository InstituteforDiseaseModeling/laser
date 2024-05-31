// update_ages.c

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include <immintrin.h>

#define SIMD_WIDTH 8     // AVX2 processes 8 integers at a time
unsigned recovered_counter = 0;

extern "C" {

/**
 * Update ages of individuals within a specified range by incrementing ages for non-negative values.
 *
 * This function increments the ages of individuals within a specified range by a constant value representing
 * one day. The function iterates over a range of indices in the `ages` array, starting from `start_idx` and
 * ending at `stop_idx` (inclusive). For each index `i` within the specified range, if the age value `ages[i]`
 * is non-negative, it is incremented by the constant value `one_day`, which represents one day in units of years.
 *
 * @param start_idx The index indicating the start of the range to update (inclusive).
 * @param stop_idx The index indicating the end of the range to update (inclusive).
 * @param ages Pointer to the array containing ages of individuals.
 *             The ages are expected to be in units of years.
 *             The array is modified in place.
 */
const float one_day = 1.0f/365.0f;
void update_ages_vanilla(unsigned long int start_idx, unsigned long int stop_idx, float *ages) {
    //printf( "%s: from %d to %d.\n", __FUNCTION__, start_idx, stop_idx );
    #pragma omp parallel for
    for (size_t i = start_idx; i <= stop_idx; i++) {
        if( ages[i] >= 0 )
        {
            ages[i] += one_day;
        }
    }
}

// avxv2
void update_ages(unsigned long int start_idx, unsigned long int stop_idx, float *ages) {
    #pragma omp parallel for
    for (size_t i = start_idx; i <= stop_idx; i += 8) {
        // Load age values into SIMD registers
        __m256 ages_vec = _mm256_loadu_ps(&ages[i]);

        // Create a vector with the one_day value replicated 8 times
        __m256 one_day_vec = _mm256_set1_ps(one_day);

        // Mask for elements greater than or equal to zero
        __m256 mask = _mm256_cmp_ps(ages_vec, _mm256_setzero_ps(), _CMP_GE_OQ);

        // Increment age values by one_day for elements greater than or equal to zero
        ages_vec = _mm256_blendv_ps(ages_vec, _mm256_add_ps(ages_vec, one_day_vec), mask);

        // Store the result back to memory
        _mm256_storeu_ps(&ages[i], ages_vec);
    }
}

/*
 * Progress all infections. Collect the indexes of those who recover. 
 * Assume recovered_idxs is pre-allocated to same size as infecteds.
 */
size_t progress_infections(
    int start_idx,
    int end_idx,
    unsigned char * infection_timer,
    unsigned char * incubation_timer,
    bool* infected,
    signed char * immunity_timer,
    bool* immunity
    //, uint32_t * recovered_idxs
) {
    recovered_counter = 0;

    // Allocate a 2D vector to store thread-local buffers
    //std::vector<std::vector<int>> thread_local_buffers(omp_get_max_threads());

#pragma omp parallel
    {
        //int thread_id = omp_get_thread_num();
        //std::vector<int> &local_buffer = thread_local_buffers[thread_id];

#pragma omp for
        for (unsigned long int i = start_idx; i <= end_idx; ++i) {
            if (incubation_timer[i] >= 1) {
                incubation_timer[i]--;
            }

            // Infection timer: decrement for each infected person
            if (infection_timer[i] >= 1) {
                infection_timer[i]--;

                // Some people clear
                if (infection_timer[i] == 0) {
                    infected[i] = 0;

                    // Recovereds gain immunity
                    immunity_timer[i] = -1;
                    immunity[i] = true;

                    //local_buffer.push_back(i);
                }
            }
        }
    }

    /*
    // Accumulate results from thread-local buffers
    int global_counter = 0;
    for (const auto &buffer : thread_local_buffers) {
        for (int idx : buffer) {
            recovered_idxs[global_counter++] = idx;
        }
    }
    recovered_counter = global_counter;
    */

    return recovered_counter;
}

void progress_immunities(
    int start_idx,
    int end_idx,
    signed char * immunity_timer,
    bool* immunity
) {
    #pragma omp parallel for
    for (int i = start_idx; i <= end_idx; ++i) {
        if( immunity[i] && immunity_timer[i] > 0 )
        {
            immunity_timer[i]--;
            if( immunity_timer[i] == 0 )
            {
                immunity[i] = false;
                //printf( "New Susceptible.\n" );
            }
        }    
    }
}


/*
 Calculate new infections based on exposed individuals and susceptible fractions.

Parameters:
- start_idx (int): The starting index of the range of individuals to process.
- end_idx (int): The ending index of the range of individuals to process. Expected to be mix of S, E and I.
- num_nodes (int): The total number of nodes in the system.
- node (uint32_t*): Array containing node identifiers.
- incubation_timers (unsigned char*): Array containing the incubation timers for each node.
- infected_fractions (float*): Array containing the fractions of infected individuals for each node.
- susceptible_fractions (float*): Array containing the fractions of susceptible individuals for each node.
- totals (uint32_t*): Array containing the total population for each node.
- new_infs_out (uint32_t*): Output array to store the calculated new infections for each node.
- base_inf (float): Base infectivity factor.

Returns:
None

Description:
Calculate the new infections for each node based on the infectious individuals, susceptible fractions, and infectivity.  Updates the new_infs_out array with the calculated number of new infections for each node.
*/
void calculate_new_infections(
    int start_idx, 
    int end_idx,
    int num_nodes,
    uint32_t * node,
    unsigned char  * incubation_timers,
    uint32_t * infected_counts,
    uint32_t * susceptible_counts,
    uint32_t * totals, // total node populations
    uint32_t * new_infs_out, // output
    float base_inf // another input :(
) {
    // We need number of infected not incubating
    unsigned int exposed_counts_by_bin[ num_nodes ];
    memset( exposed_counts_by_bin, 0, sizeof(exposed_counts_by_bin) );

    // We are not yet counting E in our regular report, so we have to count them here.
    // Is that 'expensive'? Not sure yet.
    //#pragma omp parallel for
    for (int i = start_idx; i <= end_idx; ++i) {
        if( incubation_timers[i] >= 1 ) {
            exposed_counts_by_bin[ node[ i ] ] ++;
            //printf( "DEBUG: exposed_counts_by_bin[ %d ] = %d.\n", i, exposed_counts_by_bin[node[i]] );
            //printf( "DEBUG: incubation_timers[ %d ] = %d.\n", i, incubation_timers[i] );
        }
    }

    // new infections = Infected frac * infectivity * susceptible frac * pop
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; ++i) {
        if( exposed_counts_by_bin[ i ] > infected_counts[ i ] )
        {
            printf( "ERROR: Exposed should never be > infection.\n" );
            printf( "node = %d, exposed = %d, infected = %d.\n", i, exposed_counts_by_bin[ i ], infected_counts[ i ] );
            //printf( "node = %d, exposed = %d, infected = %f.\n", i, exposed_counts_by_bin[ i ]*totals[i], infected_counts[ i ]*totals[i] );
            exposed_counts_by_bin[ i ] = infected_counts[ i ]; // HACK: Maybe an exposed count is dead?
            //abort();
        }
        float infectious_count = infected_counts[ i ] - exposed_counts_by_bin[ i ];
        float foi = infectious_count * base_inf;
        new_infs_out[ i ] = (int)round( foi * susceptible_counts[ i ] / totals[i] );
        //printf( "DEBUG: new infs[node=%d] = infected_counts(%d) * base_inf(%f) * susceptible_counts(%d) / pop(%d) = %d.\n",
               //i, infected_counts[i], base_inf, susceptible_counts[i], totals[i], new_infs_out[i] );
    }
}

void handle_new_infections(
    unsigned long int start_idx,
    unsigned long int end_idx,
    int node,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    unsigned char  * incubation_timer,
    unsigned char  * infection_timer,
    int new_infections,
    int * new_infection_idxs_out,
    int num_eligible_agents
) {
    //printf( "DEBUG: hni: creating %d new infections in node %d from %d susceptibles.\n", new_infections, node, num_eligible_agents );
    if( new_infections == 0 )
    {
        return;
    }
    assert( new_infections > 0 );
    if( num_eligible_agents == 0 )
    {
        return;
    }
    if( end_idx < start_idx ) {
        printf( "ERROR: start_idx (%ld) is not < end_idx (%ld).\n", start_idx, end_idx );
        return;
    }

    if( num_eligible_agents > 0 ) {
        unsigned long int num_agents = end_idx-start_idx+1;
        int selected_indices[num_eligible_agents]; // store idxs of selected for infection
                                                   // these should be zeroed out first.
        int count = 0;

        for (unsigned long int i = start_idx; i <= end_idx; i++) {
            if( !infected[i] && !immunity[i] && agent_node[i] == node ) {
                // Found eligible (susceptible) in group of S & Is & maybe Es and Rs
                unsigned long int selected_idx = i;
                selected_indices[count++] = selected_idx;
                //printf( "DEBUG: selected_indices[%d] = %ld.\n", count-1, selected_idx );
                if( count == num_eligible_agents ) {
                    // Note that we saw a bug where sometimes more agents were found to satisfy our sus condition than the value passed it! TBD
                    break;
                }
            }
        }
        if( count < num_eligible_agents )
        {
            printf( "ERROR: count = %d, num_eligible_agents = %d.\n", count, num_eligible_agents  );
            abort();
        }
        if( count == 0 ) {
            printf( "WARNING: Found no susceptibles for some reason. Not infecting anyone.\n" );
            abort();
            return;
        }
        //num_eligible_agents = count; // ask me about this.

        int i, step, selected_count = 0;
        // Calculate the step size
        if (new_infections >= num_eligible_agents) {
            step = 1; // If we need to select all elements or more, select each one
        } else {
            step = num_eligible_agents/new_infections; // If we need to select less than N, calculate step size
        }
        //printf( "Selecting %d new infectees by skipping through %d candidates %d at a time.\n", new_infections, num_eligible_agents, step );
        for (i = 0; i < num_eligible_agents && selected_count < new_infections; i += step) {
            unsigned long int selected_id = selected_indices[i];
            //assert( selected_id > 0 ); // this can be true
            //printf( "DEBUG: Checking if selected_id (%ld) is >= start_idx(%ld).\n", selected_id, start_idx );
            assert( selected_id >= start_idx );
            //printf( "DEBUG: Checking if selected_id (%ld) is <= end_idx(%ld).\n", selected_id, end_idx );
            assert( selected_id <= end_idx );
            //printf( "Infecting index=%d.\n", selected_id );
            infected[selected_id] = true;
            incubation_timer[selected_id] = 7;
            infection_timer[selected_id] = 14 + rand() % 2;
            new_infection_idxs_out[ selected_count++ ] = selected_id;
        }
    }
}

void handle_new_infections_mp(
    unsigned long int start_idx,
    unsigned long int end_idx,
    unsigned int num_nodes,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    unsigned char  * incubation_timer,
    unsigned char  * infection_timer,
    int * new_infections_array,
    int * num_eligible_agents_array
) {
    //printf( "handle_new_infections_mp: start_idx=%ld, end_idx=%ld.\n", start_idx, end_idx );
    std::unordered_map<int, std::vector<int>> node2sus;

#pragma omp parallel
    {
        // Thread-local buffers to collect susceptible indices by node
        std::unordered_map<int, std::vector<int>> local_node2sus;

#pragma omp for nowait
        for (unsigned long int i = start_idx; i <= end_idx; ++i) {
            if (!infected[i] && !immunity[i]) {
                int node = agent_node[i];
                local_node2sus[node].push_back(i);
                //printf( "Found susceptible in node %d\n", node );
            }
        }

#pragma omp critical
        {
            // Accumulate the local buffers into the global map
            for (const auto &pair : local_node2sus) {
                int node = pair.first;
                if (node2sus.find(node) == node2sus.end()) {
                    node2sus[node] = pair.second;
                } else {
                    node2sus[node].insert(node2sus[node].end(), pair.second.begin(), pair.second.end());
                }
            }
        }
    }

#pragma omp parallel for
    for (unsigned int node = 0; node < num_nodes; ++node) {
        unsigned int new_infections = new_infections_array[node];
        unsigned int num_eligible_agents = num_eligible_agents_array[node];
        //printf( "Node=%d, new_infections=%d, num_eligible_agents=%d\n", node, new_infections, num_eligible_agents );

        if (new_infections > 0 && num_eligible_agents > 0 && node2sus.find(node) != node2sus.end()) {
            std::vector<int> &susceptible_indices = node2sus[node];
            int num_susceptible = susceptible_indices.size();
            int step = (new_infections >= num_susceptible) ? 1 : num_susceptible / new_infections;

            for (int i = 0, selected_count = 0; i < num_susceptible && selected_count < new_infections; i += step) {
                unsigned long int selected_id = susceptible_indices[i];
                //printf( "Infecting %ld\n", selected_id );
                infected[selected_id] = true;
                incubation_timer[selected_id] = 7;
                infection_timer[selected_id] = 14 + rand() % 2;
                selected_count++;
            }
        }
    }
}

typedef struct {
    int node_id;
    // Add any other arguments needed by the worker function here
    unsigned long int start_idx;
    unsigned long int end_idx;
    uint32_t * agent_node;
    bool * infected;
    bool * immunity;
    unsigned char  * incubation_timer;
    unsigned char  * infection_timer;
    int new_infections;
    int * new_infection_idxs_out;
    int num_eligible_agents;
} ThreadArgs;


void* worker(void* arg) {
    // Unpack arg struct and call existing function that does the nodewise work.
    ThreadArgs* args = (ThreadArgs*)arg;

    if( args->new_infections == 0 )
        return NULL;
    if( args->num_eligible_agents == 0 )
        return NULL;

    int node = args->node_id; // Retrieve the thread ID
    /*
    printf( "worker: node = %d.\n", node );
    printf( "args->start_idx = %d.\n", args->start_idx );
    printf( "args->end_idx = %d.\n", args->end_idx );
    printf( "args->agent_node[0] = %d.\n", args->agent_node[0] );
    printf( "args->infected[0] = %d.\n", args->infected[0] );
    printf( "args->immunity[0] = %d.\n", args->immunity[0] );
    printf( "args->incubation_timer[0] = %d.\n", args->incubation_timer[0] );
    printf( "args->infection_timer[0] = %d.\n", args->infection_timer[0] );
    printf( "args->new_infections[0] = %d.\n", args->new_infections );
    printf( "args->new_infection_idxs_out[0] = %d.\n", args->new_infection_idxs_out[0] );
    printf( "args->num_eligible_agents[0] = %d.\n", args->num_eligible_agents );
    */
    // Call the worker function with the provided arguments
    if( args->new_infections == 0 ) {
        return NULL;
    }
    handle_new_infections(
        args->start_idx,
        args->end_idx,
        node, // Pass the thread ID
        args->agent_node,
        args->infected,
        args->immunity,
        args->incubation_timer,
        args->infection_timer,
        args->new_infections,
        args->new_infection_idxs_out,
        args->num_eligible_agents
    );
    
    return NULL;
}

// Function to calculate the effective index
int calculate_effective_index(int idx, int* new_infections) {
    int start_idx = 0;
    for (int i = 0; i < idx; i++) {
        start_idx += new_infections[i];
    }
    return start_idx;
}

void handle_new_infections_threaded(
    unsigned long int start_idx,
    unsigned long int end_idx,
    unsigned int num_nodes,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    unsigned char  * incubation_timer,
    unsigned char  * infection_timer,
    int * new_infections,
    int * new_infection_idxs_out,
    int * num_eligible_agents
) {
    if( end_idx < start_idx ) {
        printf( "ERROR: start_idx (%ld) is not < end_idx (%ld).\n", start_idx, end_idx );
        return;
    }

    unsigned long int num_threads = num_nodes;
    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    for (int i = 0; i < num_threads; i++) {
        /*
        if( new_infections[ i ] == 0 )
            continue;
        if( num_eligible_agents[ i ] == 0 )
            continue;
        */
        if( new_infections[ i ] > num_eligible_agents[ i ] )
        {
            //printf( "WARNING: Asked for %d new infections in node %d but only have %d susceptibles.\n", new_infections[ i ], i, num_eligible_agents[ i ] );
            new_infections[ i ] = num_eligible_agents[ i ];
        }
        //assert( new_infections[ i ] <= num_eligible_agents[ i ] );
        // We have to be a bit clever here with the index of new_infection_idxs_out.
        // We have a single array that's going to be written to by all the threads.
        // They all need to write to different parts, no clobbering.
        // If we are expecting 10, 4, 7, and 1 new infections, we need to pass 
        // 0, 10, 14, 21, and 22 as the start of the array to each node/thread.
        thread_args[i].node_id = i;
        thread_args[i].start_idx = start_idx;
        thread_args[i].end_idx = end_idx;
        thread_args[i].agent_node = agent_node;
        thread_args[i].infected = infected;
        thread_args[i].immunity = immunity;
        thread_args[i].incubation_timer = incubation_timer;
        thread_args[i].infection_timer = infection_timer;
        thread_args[i].new_infections = new_infections[ i ];
        int start_idx = calculate_effective_index(i, new_infections);
        thread_args[i].new_infection_idxs_out = &new_infection_idxs_out[ start_idx ];
        thread_args[i].num_eligible_agents = num_eligible_agents[ i ];
        // Initialize other arguments for the worker function

        pthread_create(&threads[i], NULL, worker, (void*)&thread_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/*
 Collects and aggregates statistical data on infection and immunity status
 of a population of agents within a specified range of indices.

 This function iterates over a subset of agents defined by the range
 [start_idx, eula_idx] and collects data on their infection and immunity
 status. It calculates and aggregates counts of susceptible, infected,
 and recovered agents for each unique node in the population.

 - num_agents: The total number of agents in the population.
 - start_idx: The starting index of the subset of agents to process.
 - eula_idx: The ending index of the subset of agents to process.
 - node: An array containing the node IDs of each agent.
 - infected: An array indicating whether each agent is infected.
 - immunity: An array indicating whether each agent is immune.
 - age: An array containing the age of each agent.
 - expected_lifespan: An array containing the expected lifespan of each agent.
 - infection_count: An array to store the count of infected agents per node.
 - susceptible_count: An array to store the count of susceptible agents per node.
 - recovered_count: An array to store the count of recovered agents per node.
*/
void collect_report_vanilla( 
    int num_agents,
    int start_idx,
    int eula_idx,
    uint32_t * node,
    bool * infected,
    bool * immunity,
    float * age,
    float * expected_lifespan, // so we can not count dead people
    uint32_t * infection_count,
    uint32_t * susceptible_count,
    uint32_t * recovered_count
)
{
    //printf( "%s called w/ num_agents = %d, start_idx = %d, eula_idx = %d.\n", __FUNCTION__, num_agents, start_idx, eula_idx );
    for (int i = start_idx; i <= eula_idx; ++i) {
        if( node[i] >= 0 )
        {
            int node_id = node[i];
            if( age[ i ] < expected_lifespan[ i ] ) {
                if( infected[ i ] ) {
                    infection_count[ node_id ] ++;
                    //printf( "Incrementing I count for node %d = %d from idx %d.\n", node_id, infection_count[ node_id ], i );
                } else if( immunity[ i ] ) {
                    recovered_count[ node_id ] ++;
                    //printf( "Incrementing R count for node %d = %d from idx %d.\n", node_id, recovered_count[ node_id ], i );
                } else {
                    susceptible_count[ node_id ] ++;
                    //printf( "Incrementing S count for node %d = %d from idx %d.\n", node_id, susceptible_count[ node_id ], i );
                }
            }
        }
    }
    for (int i = eula_idx; i < num_agents; ++i) {
        int node_id = node[i];
        if( age[ i ] < expected_lifespan[ i ] ) {
            recovered_count[ node_id ]++;
        }
    }
}

void collect_report(
    int num_agents,
    int start_idx,
    int eula_idx,
    uint32_t * node,
    bool * infected,
    bool * immunity,
    float * age,
    float * expected_lifespan, // so we can not count dead people
    uint32_t * infection_count,
    uint32_t * susceptible_count,
    uint32_t * recovered_count
)
{
    unsigned int num_nodes=954; // pass this in
    #pragma omp parallel
    {
        // Thread-local buffers
        int *local_infection_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_recovered_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_susceptible_count = (int*) calloc(num_nodes, sizeof(int));

        #pragma omp for nowait
        for (int i = start_idx; i <= eula_idx; ++i) {
            if (node[i] >= 0) {
                int node_id = node[i];
                if (age[i] < expected_lifespan[i]) {
                    if (infected[i]) {
                        local_infection_count[node_id]++;
                    } else if (immunity[i]) {
                        local_recovered_count[node_id]++;
                    } else {
                        local_susceptible_count[node_id]++;
                    }
                }
            }
        }

        #pragma omp for nowait
        for (int i = eula_idx; i < num_agents; ++i) {
            int node_id = node[i];
            if (age[i] < expected_lifespan[i]) {
                local_recovered_count[node_id]++;
            }
        }

        // Combine local counts into global counts
        #pragma omp critical
        {
            for (int j = 0; j < num_nodes; ++j) {
                infection_count[j] += local_infection_count[j];
                recovered_count[j] += local_recovered_count[j];
                susceptible_count[j] += local_susceptible_count[j];
            }
        }

        // Free local buffers
        free(local_infection_count);
        free(local_recovered_count);
        free(local_susceptible_count);
    }
}

#if 0
const int max_node_id = 953;
void migrate( int num_agents, int start_idx, int end_idx, bool * infected, uint32_t * node ) {
    // This is just a very simplistic one-way linear type of infection migration
    // I prefer to hard code a few values for this function rather than add parameters
    // since it's most a test function.
    int fraction = (int)(0.02*1000); // this fraction of infecteds migrate
    unsigned long int counter = 0;
    #pragma omp parallel for
    for (int i = start_idx; i < num_agents; ++i) {
        if( i != end_idx ) {
            if( infected[ i ] && rand()%1000 < fraction )
            {
                if( node[ i ] > 0 )
                {
                    node[ i ] --;
                }
                else
                {
                    node[ i ] = max_node_id; // this should be param
                }
            }
        }
    }
}
#endif

//////////////////////////////////////////////////////////////////////
// Helper function to select destination
static inline int select_destination(int source_index, double random_draw, double *attraction_probs, int num_locations) {
    for (int i = 0; i < num_locations; ++i) {
        if (attraction_probs[source_index * num_locations + i] > random_draw) {
            return i;
        }
    }
    return num_locations - 1;  // Fallback to the last index if none found (shouldn't normally happen)
}

// The main function to be called from Python
void migrate(
    int start_idx,
    int end_idx,
    bool *infected,
    unsigned char  * incubation_timer,
    int *data_node,
    int *data_home_node,
    double *attraction_probs,
    double migration_fraction,
    int num_locations
) {
    int num_agents = end_idx - start_idx;
    //printf( "[migrate] num_agents = %d\n", num_agents );

    #pragma omp parallel for
    for (int i = start_idx; i <= end_idx; ++i) {
        if (infected[i] && incubation_timer[i] <= 0) {
            double rand_value = (double)rand() / RAND_MAX;
            if (rand_value < migration_fraction) {
                int source_index = data_node[i];
                double random_draw = (double)rand() / RAND_MAX;
                int destination_index = -1;

                // Find the destination index based on attraction probabilities
                for (int j = 0; j < num_locations; ++j) {
                    if (attraction_probs[source_index * num_locations + j] > random_draw) {
                        destination_index = j;
                        break;
                    }
                }

                if (destination_index != -1) {
                    data_home_node[i] = data_node[i];
                    //printf( "DEBUG: Migrating individual %d from %d to %d.\n", i, data_node[i], destination_index );
                    data_node[i] = destination_index;
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////

unsigned long int campaign(
    int num_agents,
    int start_idx,
    float coverage,
    int campaign_node,
    bool *immunity,
    signed char  *immunity_timer,
    float *age,
    int *node
)
{
    // We have in mind a vaccination campaign to a subset of the population based on age, in a particular node, at
    // a particular coverage level.
    // The intervention effect will be to make them permanently immune.
    // Create a boolean mask for the conditions specified in the WHERE clause
    unsigned long int report_counter = 0;
    // printf( "DEBUG: Looking through %d susceptible agents in node %d under age %f with coverage %f to give immunity.\n", num_agents, campaign_node, 16.0f, coverage );
    for (int i = start_idx; i < num_agents; ++i) {
        if( age[i] < 16 &&
            age[i] > 0 &&
            node[i] == campaign_node &&
            immunity[i] == false &&
            rand()%100 < 100*coverage
        )
        {
            //printf( "Changing value of immunity at index %d.\n", i );
            immunity[i] = true;
            immunity_timer[i] = -1;
            report_counter ++;
        }
    }
    return report_counter;
}

unsigned long int ria(
    int num_agents,
    int start_idx, // to count backwards
    float coverage,
    int campaign_node,
    bool *immunity,
    signed char  *immunity_timer,
    float *age,
    int *node,
    int *immunized_indices
)
{
    // We have in mind a vaccination campaign to a fraction of the population turning 9mo, in a particular node, at
    // a particular coverage level.
    // The intervention effect will be to make them permanently immune.
    // Create a boolean mask for the conditions specified in the WHERE clause
    unsigned long int report_counter = 0; // not returned for now
    // printf( "DEBUG: Looking through %d susceptible agents in node %d under age %f with coverage %f to give immunity.\n", num_agents, campaign_node, 16.0f, coverage );
    unsigned long int new_idx = start_idx;
    //printf( "%s called with start_idx=%d, counting down to %d.\n", __FUNCTION__, start_idx, num_agents );
    for (int i = start_idx; i > num_agents; --i) {
        printf( "age = %f.\n", age[i] );
        // keep decrementing until we get kids younger than 0.75
        if( age[i] < 0.75 ) {
            //printf( "age of individual at idx %d = %f. Cutting out of loop.\n", i, age[i] );
            new_idx = i;
            break;
        }

        float upper_bound = 0.75+30/365.; // We want to be able to hunt from oldest in pop down to "9 month old" 
                                          // without vaxxing them all. But then later we want to be able to grab
                                          // everyone who aged into 9months while we were away and vax them. Tricky.
        if( age[i] > upper_bound ) {
            //printf( "Too old. Keep counting down to find '9-month-olds'.\n" );
            continue; // keep counting down
        }

        if( node[i] == campaign_node &&
            immunity[i] == false &&
            rand()%100 < 0.75*coverage
        )
        {
            printf( "Changing value of immunity at index %d.\n", i );
            immunity[i] = true;
            immunity_timer[i] = -1;
            immunized_indices[ report_counter ++ ] = i;
        }
        else
        {
            printf( "Didn't match immunity and coverage filter.\n" );
        }
    }
    /*if( report_counter > 0 ) {
        printf( "Vaccinated %d 9-month-olds starting at idx %d and ending up at %d.\n", report_counter, start_idx, new_idx );
    }*/
    return new_idx;
}

void reconstitute(
    int start_idx,
    int num_new_babies,
    int* new_nodes,
    int *node,
    float *age
) {
    //printf( "%s: num_new_babies = %d\n", __FUNCTION__, num_new_babies );
    int counter = 0;
    for (int i = start_idx; i > 0; --i) {
        if( age[i] < 0 ) {
            node[i] = new_nodes[ counter ];
            age[i] = 0;
            counter ++;
            if( counter == num_new_babies ) {
                return;
            }
        }
        else {
            printf( "ERROR: Next U (idx=%d) wasn't the right age (%f) for some reason!.\n", i, age[i] );
        }
    }
    printf( "ERROR: We ran out of open slots for new babies!" );
    abort();
}

double random_double() {
    return (double) rand() / RAND_MAX;
}

// Function to generate a binomial random variable
int binomial(int n, double p) {
    int successes = 0;
    for (int i = 0; i < n; ++i) {
        if (random_double() < p) {
            successes++;
        }
    }
    return successes;
}

/*
 * Need access to the eula map/dict. Probably should pass in the sorted values as an array
 */
void progress_natural_mortality_binned(
    int* eula, // sorted values as an array
    int num_nodes,
    int num_age_bins,  // size of eula array
    float* probs,
    int timesteps_elapsed // how many timesteps are we covering
) {
    // Iterate over nodes and age bins
    for (int node = 0; node < num_nodes; ++node) {
        for (int age_bin = 0; age_bin < num_age_bins; ++age_bin) {
            // Compute expected deaths
            float prob = probs[age_bin]; // Implement this function as needed
            int count = eula[node * num_age_bins + age_bin];
            int expected_deaths = 0;
            for (int i = 0; i < timesteps_elapsed; ++i) {
                expected_deaths += binomial(count, prob); // Implement binomial function as needed
            }
            eula[node * num_age_bins + age_bin] -= expected_deaths;
        }
    }
}

/////////////////////////////
// DEPRECATED
/////////////////////////////

float logistic_density_fn(float x) {
    float L = 4.5; // Maximum value
    float x0 = 250000.0; // Midpoint
    float k = 0.0001; // Steepness parameter
    float ret = 0.5 + (L / (1.0 + exp(-k * (x - x0))));
    //printf( "ccs multiplier = %f.\n", ret );
    return ret;
}

// Function to generate a random number of new infections
int generate_new_infections(int N, double P) {
    // Generate a random number of new infections from a binomial distribution
    int new_infections = 0;
    for (int i = 0; i < N; i++) {
        double rand_num = (double)rand() / RAND_MAX;  // Generate a random number between 0 and 1
        if (rand_num < P) {  // Probability of infection
            new_infections++;
        }
    }
    //printf( "generate_new_infections returning %d for num sus = %d and prob = %f.\n", new_infections, N, P );
    return new_infections;
}


}
