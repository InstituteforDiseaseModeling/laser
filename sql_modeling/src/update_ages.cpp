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

const float one_day = 1.0f/365.0f;
static std::unordered_map<int,std::deque<int>> infection_queue_map;
static std::unordered_map<int,std::deque<int>> incubation_queue_map;
std::mutex map_mtx;

extern "C" {

void init_maps(
    size_t n,
    int start_idx,
    const bool * infected,
    const float * infection_timer
) {
    for (int i = start_idx; i < n; ++i) {
        if( infected[ i ] ) {
            infection_queue_map[ int(infection_timer[ i ]) ].push_back( i );
            //printf( "%d: infection_queue_map[ %d ].size() = %lu.\n", __LINE__, int(infection_timer[ i ] ), infection_queue_map[ int(infection_timer[ i ]) ].size() );
            incubation_queue_map[ 3 ].push_back( i );
            //printf( "%d: incubation_queue_map[ 3 ].size() = %lu.\n", __LINE__, incubation_queue_map[ 3 ].size() );
        }
    }
}

void update_ages(size_t length, float *ages) {
    for (size_t i = 0; i < length; i++) {
        if( ages[i] < 0 )
        {
            continue;
        }
        ages[i] += one_day;
    }
}

void progress_infections2(
    int n,
    int start_idx,
    float* infection_timer,
    float* incubation_timer,
    bool* infected,
    float* immunity_timer,
    bool* immunity,
    int timestep
)
{
    if (incubation_queue_map.find(timestep) != incubation_queue_map.end()) {
        std::deque<int>& activators = incubation_queue_map[timestep];
        for (int idx : activators) {
            incubation_timer[idx] = 0;
        }
        incubation_queue_map.erase(timestep);
        //incubation_queue_map[timestep].clear();
    }
 
    if (infection_queue_map.find(timestep) != infection_queue_map.end()) {
        std::deque<int>& recovereds = infection_queue_map[timestep];
        for (int idx : recovereds) {
            infection_timer[idx] = 0;
            infected[idx] = false;
            immunity_timer[idx] = -1;
            immunity[idx] = true;
        }
        infection_queue_map.erase(timestep);
        //infection_queue_map[timestep].clear();
    }
    //printf( "%d: infection_queue_map[ %d ].size() = %lu.\n", __LINE__, timestep, infection_queue_map[ timestep ].size() );
    //printf( "%d: incubation_queue_map[ %d ].size() = %lu.\n", __LINE__, timestep, incubation_queue_map[ timestep ].size() );
}

/*
 * Progress all infections. Collect the indexes of those who recover. 
 * Assume recovered_idxs is pre-allocated to same size as infecteds.
 */
size_t progress_infections(
    int n,
    int start_idx,
    float* infection_timer,
    float* incubation_timer,
    bool* infected,
    float* immunity_timer,
    bool* immunity,
    int* node,
    int* recovered_idxs
) {
    unsigned int activators = 0;

    std::deque<int> recovereds;
    for (int i = start_idx; i < n; ++i) {
        if (infected[i] ) {
            // Incubation timer: decrement for each person
            if (incubation_timer[i] >= 1) {
                incubation_timer[i] --;
                /*if( incubation_timer[i] == 0 )
                {
                    //printf( "Individual %d activating; incubation_timer= %f.\n", i, incubation_timer[i] );
                    activators++;
                }*/
            }

            // Infection timer: decrement for each infected person
            if (infection_timer[i] >= 1) {
                infection_timer[i] --;


                // Some people clear
                if ( infection_timer[i] == 0) {
                    infected[i] = 0;

                    // Recovereds gain immunity
                    //immunity_timer[i] = rand() % (30) + 10;  // Random integer between 10 and 40
                    // Make immunity permanent for now; we'll want this configurable at some point
                    immunity_timer[i] = -1;
                    immunity[i] = 1;
                    recovereds.push_back( i );
                }
            }
        }
    }
    //printf( "%d activators, %d recovereds.\n", activators, recovereds );
    if( recovereds.size() > 0 ) {
        //printf( "Returning %lu recovered indexes; first one is %d.\n", recovereds.size(), recovereds[0] );
        std::copy(recovereds.begin(), recovereds.end(), recovered_idxs);
    }
    return recovereds.size();
}

void progress_immunities(int n, int start_idx, float* immunity_timer, bool* immunity, int* node) {
    for (int i = start_idx; i < n; ++i) {
        if( immunity[i] && immunity_timer[i] > 0 )
        {
            immunity_timer[i]--;
            if( immunity_timer[i] == 0 )
            {
                immunity[i] = false;
            }
        }    
    }
}

// Dang, this one is slower than the numpy version!?!?
// maybe I need to just use the 64-bit ints and avoid the casting
void calculate_new_infections(
    int num_agents,
    int start_idx, 
    int num_nodes,
    uint32_t * node,
    float * incubation_timer,
    float * infection_counts,
    float * sus,
    float * totals,
    uint32_t * new_infs_out,
    float base_inf
) {
    // We need number of infected not incubating
    float exposed_counts_by_bin[ num_nodes ];
    memset( exposed_counts_by_bin, 0, sizeof(exposed_counts_by_bin) ); // not sure if this helps

    for (int i = start_idx; i < num_agents; ++i) {
        if( incubation_timer[i] >= 1 ) {
            exposed_counts_by_bin[ node[ i ] ] ++;
            // printf( "DEBUG: incubation_timer[ %d ] = %f.\n", i, incubation_timer[i] );
        }
    }

    // new infections = Infecteds * infectivity * susceptibles
    for (int i = 0; i < num_nodes; ++i) {
        //printf( "exposed_counts_by_bin[%d] = %f.\n", i, exposed_counts_by_bin[i] );
        exposed_counts_by_bin[ i ] /= totals[ i ];
        if( exposed_counts_by_bin[ i ] > infection_counts[ i ] )
        {
            printf( "Exposed should never be > infection.\n" );
            printf( "node = %d, exposed = %f, infected = %f.\n", i, exposed_counts_by_bin[ i ]*totals[i], infection_counts[ i ]*totals[i] );
            abort();
        }
        infection_counts[ i ] -= exposed_counts_by_bin[ i ];
        //printf( "infection_counts[%d] = %f\n", i, infection_counts[i] );
        float foi = infection_counts[ i ] * base_inf;
        //printf( "foi[%d] = %f\n", i, foi );
        new_infs_out[ i ] = (int)( foi * sus[ i ] );
        //printf( "new infs[%d] = foi(%f) * sus(%f) = %d.\n", i, foi, sus[i], new_infs_out[i] );
    }
}

void handle_new_infections(
    int num_agents,
    int start_idx,
    int node,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    float * incubation_timer,
    float * infection_timer,
    int new_infections,
    int * new_infection_idxs_out,
    int timestep
) {
    //printf( "Infect %d new people.\n", new_infections );
    //std::map< int, int > id2idxMap;
    // Allocate memory for subquery_condition array
    bool *subquery_condition = (bool*)malloc(num_agents * sizeof(bool));
    
    // Apply conditions to identify eligible agents
    for (int i = start_idx; i < num_agents; i++) {
        subquery_condition[i] = !infected[i] && !immunity[i] && agent_node[i] == node;
    }
    
    // Initialize random number generator
    srand(time(NULL)); // TBD: this should just be done once.
    
    // Count the number of eligible agents
    int num_eligible_agents = 0;
    for (int i = start_idx; i < num_agents; i++) {
        if (subquery_condition[i]) {
            num_eligible_agents++;
        }
    }
    if( num_eligible_agents == 0 ) {
        printf( "num_eligible_agents=0. Returning early.\n" );
        return;
    }
    
    // Allocate memory for selected_indices array
    int *selected_indices = (int*) malloc(num_eligible_agents * sizeof(int));
    
    // Randomly sample from eligible agents
    int count = 0;
    for (int i = start_idx; i < num_agents; i++) {
        if (subquery_condition[i]) {
            selected_indices[count++] = i;
        }
    }
    // Shuffle the selected_indices array
    for (int i = num_eligible_agents - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = selected_indices[i];
        selected_indices[i] = selected_indices[j];
        selected_indices[j] = temp;
    }

    // Update the 'infected' column based on selected indices
    int num_infections = (new_infections < num_eligible_agents) ? new_infections : num_eligible_agents;
    std::deque<int> new_incubators;
    unsigned int inc_time_idx = int(timestep + 3);
    for (int i = 0; i < num_infections; i++) {
        unsigned int selected_id = selected_indices[i];
        infected[selected_id] = true;
        incubation_timer[selected_id] = 3; 
        infection_timer[selected_id] = rand() % (10) + 7; // Random integer between 4 and 14;
        new_infection_idxs_out[ i ] = selected_id;

        /*
        // maps code
        unsigned int recovery_time = int(timestep+infection_timer[selected_id]);
        {
        std::lock_guard<std::mutex> lock(map_mtx);
        infection_queue_map[ recovery_time ].push_back( selected_id );
        }
        {
        std::lock_guard<std::mutex> lock(map_mtx);
        incubation_queue_map[ inc_time_idx ].push_back( selected_id );
        }
        //printf( "%d: infection_queue_map[ %d ].size() = %lu.\n", __LINE__, recovery_time, infection_queue_map[recovery_time].size() );
        //printf( "%d: incubation_queue_map[ %d ].size() = %lu.\n", __LINE__, inc_time_idx, incubation_queue_map[inc_time_idx].size() );
        */
    }

    // Free dynamically allocated memory
    free(subquery_condition);
    free(selected_indices);
}

void migrate( int num_agents, int start_idx, int end_idx, bool * infected, uint32_t * node ) {
    // This is just a very simplistic one-way linear type of infection migration
    // I prefer to hard code a few values for this function rather than add parameters
    // since it's most a test function.
    int fraction = (int)(0.02*1000); // this fraction of infecteds migrate
    unsigned int counter = 0;
    for (int i = start_idx; i < num_agents; ++i) {
        if( i==end_idx ) {
            return;
        }
        if( infected[ i ] && rand()%1000 < fraction )
        {
            if( node[ i ] > 0 )
            {
                node[ i ] --;
            }
            else
            {
                node[ i ] = 59; // this should be param
            }
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
    uint32_t * infection_count,
    uint32_t * susceptible_count,
    uint32_t * recovered_count
)
{
    //printf( "%s called w/ num_agents = %d, start_idx = %d, eula_idx = %d.\n", __FUNCTION__, num_agents, start_idx, eula_idx );
    //for (int i = start_idx; i < eula_idx; ++i) {
    for (int i = start_idx; i < num_agents; ++i) {
        //printf( "i=%d\n", i );
        if( node[i] < 0 ) {
            continue;
        }
        int node_id = node[i];
        //printf( "node_id=%d\n", node_id );
        /*if ( node_id == (uint32_t)-1 ) {
            printf( "node_id is (uint32_t)-1\n" );
            break;
        } else if ( node_id < 0 ) {
            printf( "node_id is negative\n" );
            break;
        } else if ( node_id == 1091567616 ) {
            printf( "node_id is 1091567616\n" );
            break;
        }*/
        if( infected[ i ] ) {
            infection_count[ node_id ]+=1;
            //printf( "Incrementing I count for node %d = %d from idx %d.\n", node_id, infection_count[ node_id ], i );
        } else if( immunity[ i ] ) {
            recovered_count[ node_id ]+=1;
            //printf( "Incrementing R count for node %d = %d from idx %d.\n", node_id, recovered_count[ node_id ], i );
        } else {
            susceptible_count[ node_id ]+=1;
            //printf( "Incrementing S count for node %d = %d from idx %d.\n", node_id, susceptible_count[ node_id ], i );
        }
    }
    /*for (int i = eula_idx; i < num_agents; ++i) {
        int node_id = node[i];
        recovered_count[ node_id ]++;
    }*/
}

unsigned int campaign(
    int num_agents,
    int start_idx,
    float coverage,
    int campaign_node,
    bool *immunity,
    float *immunity_timer,
    float *age,
    int *node
)
{
    // We have in mind a vaccination campaign to a subset of the population based on age, in a particular node, at
    // a particular coverage level.
    // The intervention effect will be to make them permanently immune.
    // Create a boolean mask for the conditions specified in the WHERE clause
    unsigned int report_counter = 0;
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

unsigned int ria(
    int num_agents,
    int start_idx, // to count backwards
    float coverage,
    int campaign_node,
    bool *immunity,
    float *immunity_timer,
    float *age,
    int *node
)
{
    // We have in mind a vaccination campaign to a fraction of the population turning 9mo, in a particular node, at
    // a particular coverage level.
    // The intervention effect will be to make them permanently immune.
    // Create a boolean mask for the conditions specified in the WHERE clause
    unsigned int report_counter = 0; // not returned for now
    // printf( "DEBUG: Looking through %d susceptible agents in node %d under age %f with coverage %f to give immunity.\n", num_agents, campaign_node, 16.0f, coverage );
    unsigned int new_idx = start_idx;
    //printf( "%s called with start_idx=%d, counting down to %d.\n", __FUNCTION__, start_idx, num_agents );
    for (int i = start_idx; i > num_agents; --i) {
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
            continue; // keep counting down
        }

        if( node[i] == campaign_node &&
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
    /*if( report_counter > 0 ) {
        printf( "Vaccinated %d 9-month-olds starting at idx %d and ending up at %d.\n", report_counter, start_idx, new_idx );
    }*/
    return new_idx;
}

void reconstitute(
    int num_agents,
    int start_idx,
    int num_new_babies,
    int* new_nodes,
    int *node,
    float *age,
    bool *infected,
    float *incubation_timer,
    bool *immunity,
    float *immunity_timer,
    float *expected_lifespan,
    int* new_ids_out
) {
    //printf( "%s: num_new_babies = %d\n", __FUNCTION__, num_new_babies );
    int counter = 0;
    for (int i = start_idx; i > 0; --i) {
    //for (int i = num_agents; i > 0; --i) {
        if( age[i] < 0 ) {
            node[i] = new_nodes[ counter ];
            age[i] = 0;
            infected[i] = false;
            incubation_timer[i] = 0;
            immunity[i] = 0;
            immunity_timer[i] = 0;
            expected_lifespan[i] = 75;

            new_ids_out[counter] = i;
            counter ++;
            if( counter == num_new_babies ) {
                return;
            }
        }
    }
    printf( "ERROR: We ran out of open slots for new babies!" );
    abort();
}
}
