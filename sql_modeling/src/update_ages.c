// update_ages.c

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

const float one_day = 1.0f/365.0f;
void update_ages(size_t length, float *ages) {
    for (size_t i = 0; i < length; i++) {
        if( ages[i] < 0 )
        {
            continue;
        }
        ages[i] += one_day;
    }
}

void progress_infections(int n, float* infection_timer, float* incubation_timer, bool* infected, float* immunity_timer, bool* immunity) {
    unsigned int activators = 0;
    unsigned int recovereds = 0;

    for (int i = 0; i < n; ++i) {
        if (infected[i] ) {
            // Incubation timer: decrement for each person
            if (incubation_timer[i] >= 1) {
                incubation_timer[i] --;
                if( incubation_timer[i] == 0 )
                {
                    //printf( "Individual %d activating; incubation_timer= %f.\n", i, incubation_timer[i] );
                    activators++;
                }
            }

            // Infection timer: decrement for each infected person
            if (infection_timer[i] >= 1) {
                infection_timer[i] --;


                // Some people clear
                if ( infection_timer[i] == 0) {
                    recovereds ++;
                    infected[i] = 0;

                    // Recovereds gain immunity
                    //immunity_timer[i] = rand() % (30) + 10;  // Random integer between 10 and 40
                    immunity_timer[i] = -1;
                    immunity[i] = 1;
                }
            }
        }
    }
    //printf( "%d activators, %d recovereds.\n", activators, recovereds );
}

void progress_immunities(int n, float* immunity_timer, bool* immunity) {
    for (int i = 0; i < n; ++i) {
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

    for (int i = 0; i < num_agents; ++i) {
        if( node[i] < 0 ) {
            continue;
        }
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
            printf( "i = %d, exposed = %f, infected = %f, totals = %f.\n", i, exposed_counts_by_bin[ i ], infection_counts[ i ], totals[i] );
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

void handle_new_infections2(
    int num_agents,
    int node,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    float * incubation_timer,
    float * infection_timer,
    int new_infections
)
{
    // Create a boolean mask based on the conditions in the subquery
    // This kind of sucks
    bool* subquery_condition = (bool*)malloc(num_agents * sizeof(bool));
    //memset( subquery_condition, 0, sizeof(subquery_condition) );

    // Would be really nice to avoid looping over all the agents twice, especially since this is
    // being called for each node (that has new infections).
    for (int i = 0; i < num_agents; ++i) {
        if( agent_node[i] < 0 ) {
            continue;
        }
        // Not infected AND not immune (susceptible) AND in this node
        subquery_condition[i] = (infected[i]==false) && (immunity[i]==false) && (agent_node[i] == node);
    }

    // Get the indices of eligible agents using the boolean mask
    uint32_t* eligible_agents_indices = (uint32_t*)malloc(num_agents * sizeof(uint32_t));
    int count = 0;

    for (int i = 0; i < num_agents; ++i) {
        if( agent_node[i] < 0 ) {
            continue;
        }
        if (subquery_condition[i]) {
            eligible_agents_indices[count++] = i;
        }
    }
    free(subquery_condition);

    // Randomly sample 'new_infections' number of indices
    uint32_t* selected_indices = (uint32_t*)malloc(new_infections * sizeof(uint32_t));

    for (uint32_t i = 0; i < new_infections && i < count; ++i) {
        int random_index = rand() % count;
        // TBD: This needs to be change to avoid duplicates (with replacement or whatever)
        selected_indices[i] = eligible_agents_indices[random_index];
    }

    // Update the 'infected' column based on the selected indices
    for (uint32_t i = 0; i < new_infections && i < count; ++i) {
        uint32_t selected_id = selected_indices[i];
        if( infected[ selected_id ] )
        {
            printf( "ERROR: infecting already infected individual: %d. No idea how we got here. Let's try skipping for now and see how it goes.\n", selected_id );
            //abort();
        }
        else {
            infected[selected_id] = true;  // Assuming 1 represents True
            unsigned int min_infection_duration = 3;
            incubation_timer[selected_id] = min_infection_duration; 
            unsigned int max_infection_dur = min_infection_duration + 10; // 3-13
            infection_timer[selected_id] = rand() % (max_infection_dur) + min_infection_duration; // Random integer between 3 and 13;ish.
            //printf( "Infecting individual %d.\n", selected_id );
        }
    }

    // Free allocated memory
    free(eligible_agents_indices);
    free(selected_indices);
}

void handle_new_infections(
    int num_agents,
    int node,
    uint32_t * agent_node,
    bool * infected,
    bool * immunity,
    float * incubation_timer,
    float * infection_timer,
    int new_infections
) {
    //printf( "Infect %d new people.\n", new_infections );
    // Allocate memory for subquery_condition array
    bool *subquery_condition = malloc(num_agents * sizeof(bool));
    
    // Apply conditions to identify eligible agents
    for (int i = 0; i < num_agents; i++) {
        if( agent_node[i] < 0 ) {
            continue;
        }
        subquery_condition[i] = !infected[i] && !immunity[i] && agent_node[i] == node;
    }
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Count the number of eligible agents
    int num_eligible_agents = 0;
    for (int i = 0; i < num_agents; i++) {
        if( agent_node[i] < 0 ) {
            continue;
        }
        if (subquery_condition[i]) {
            num_eligible_agents++;
        }
    }
    
    // Allocate memory for selected_indices array
    int *selected_indices = malloc(num_eligible_agents * sizeof(int));
    
    // Randomly sample from eligible agents
    int count = 0;
    for (int i = 0; i < num_agents; i++) {
        if( agent_node[i] < 0 ) {
            continue;
        }
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
    for (int i = 0; i < num_infections; i++) {
        unsigned int selected_id = selected_indices[i];
        infected[selected_id] = true;
        incubation_timer[selected_id] = 3; 
        infection_timer[selected_id] = rand() % (10) + 7; // Random integer between 4 and 14;
    }

    // Free dynamically allocated memory
    free(subquery_condition);
    free(selected_indices);
}

void migrate( int num_agents, bool * infected, uint32_t * node ) {
    // This is just a very simplistic one-way linear type of infection migration
    // I prefer to hard code a few values for this function rather than add parameters
    // since it's most a test function.
    int fraction = (int)(0.02*1000); // this fraction of infecteds migrate
    unsigned int counter = 0;
    for (int i = 0; i < num_agents; ++i) {
        if( node[i] < 0 ) {
            continue;
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
    uint32_t * node,
    bool * infected,
    bool * immunity,
    uint32_t * infection_count,
    uint32_t * susceptible_count,
    uint32_t * recovered_count
)
{
    for (int i = 0; i < num_agents; ++i) {
        if( node[i] < 0 ) {
            continue;
        }
        uint32_t node_id = node[i];
        if ( node_id == (uint32_t)-1 ) {
            continue;
        }
        if( infected[ i ] ) {
            infection_count[ node_id ]+=1;
            //printf( "Adding %d to I count for node %d = %d.\n", mcw[i], node_id, infection_count[ node_id ] );
        } else if( immunity[ i ] ) {
            recovered_count[ node_id ]+=1;
            //printf( "Adding %d to R count for node %d = %d.\n", mcw[i], node_id, recovered_count[ node_id ] );
        } else {
            // HOW THE HECK AM I GETTING THE EXACT SAME NUMBER OF SUSCEPTIBLES 1 at a time AS RECOVEREDS?!?!?!?!??!??!
            susceptible_count[ node_id ]+=1;
            //printf( "Adding %d to S count for node %d = %d.\n", mcw[i], node_id, susceptible_count[ node_id ] );
        }
    }
}

unsigned int campaign(
    int num_agents,
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
    for (int i = 0; i < num_agents; ++i) {
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

void reconstitute(
    int num_agents,
    int num_new_babies,
    int* new_nodes,
    int *node,
    float *age,
    bool *infected,
    float *incubation_timer,
    bool *immunity,
    float *immunity_timer,
    float *expected_lifespan
) {
    //printf( "%s: num_new_babies = %d\n", __FUNCTION__, num_new_babies );
    int counter = 0;
    for (int i = 0; i < num_agents; ++i) {
    //for (int i = num_agents; i > 0; --i) {
        if( age[i] < 0 ) {
            node[i] = new_nodes[ counter ];
            age[i] = 0;
            infected[i] = false;
            incubation_timer[i] = 0;
            immunity[i] = 0;
            immunity_timer[i] = 0;
            expected_lifespan[i] = 75;

            counter ++;
            if( counter == num_new_babies ) {
                return;
            }
        }
    }
    printf( "ERROR: We ran out of open slots for new babies!" );
    abort();
}
