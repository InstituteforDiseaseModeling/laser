// update_ages.c

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void update_ages(double *ages, size_t length) {
    for (size_t i = 0; i < length; i++) {
        ages[i] += 1.0 / 365.0;
    }
}

void progress_infections(int n, float* infection_timer, float* incubation_timer, bool* infected, float* immunity_timer, bool* immunity) {
    for (int i = 0; i < n; ++i) {
        if (infected[i] ) {
        // Infection timer: decrement for each infected person
        if (infection_timer[i] >= 1) {
            infection_timer[i] -= 1;

        // Incubation timer: decrement for each person
        if (incubation_timer[i] >= 1) {
            incubation_timer[i] -= 1;
        }

            // Some people clear
            if ( infection_timer[i] == 0) {
                infected[i] = 0;

            // Recovereds gain immunity
                immunity_timer[i] = rand() % (41 - 10 + 1) + 10;  // Random integer between 10 and 40
                immunity[i] = 1;
            }
        }
        }
    }
}

void progress_immunities(int n, float* immunity_timer, bool* immunity) {
    for (int i = 0; i < n; ++i) {
        if( immunity[i] && immunity_timer[i] > 0 )
	{
            immunity_timer[i]--;
	    if( immunity_timer[i] == 0 )
	    {
	        immunity[i] = true;
	    }
	}	
    }
}

// Dang, this one is slower than the numpy version!?!?
// maybe I need to just use the 64-bit ints and avoid the casting
void calculate_new_infections(
    int num_agents,
    int num_nodes,
    float * incubation_timer,
    uint32_t * node,
    uint32_t * infection_counts,
    uint32_t * sus,
    uint32_t * new_infs_out
) {
    // We need number of infected not incubating
    const float base_inf = 0.0001f;
    int exposed_counts_by_bin[ num_nodes ];
    memset( exposed_counts_by_bin, 0, sizeof(exposed_counts_by_bin) ); // not sure if this helps

    for (int i = 0; i < num_agents; ++i) {
	if( incubation_timer[i] >= 1 ) {
	    exposed_counts_by_bin[ node[ i ] ] ++;
	}
    }

    // new infections = Infecteds * infectivity * susceptibles
    for (int i = 0; i < num_nodes; ++i) {
	infection_counts[ i ] -= exposed_counts_by_bin[ i ];
	float foi = infection_counts[ i ] * base_inf;
	new_infs_out[ i ] = (int)( foi * sus[ i ] );
    }
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
)
{
    // Create a boolean mask based on the conditions in the subquery
    // This kind of sucks
    bool* subquery_condition = (bool*)malloc(num_agents * sizeof(bool));

    // Would be really nice to avoid looping over all the agents twice, especially since this is
    // being called for each node (that has new infections).
    for (int i = 0; i < num_agents; ++i) {
        subquery_condition[i] = !infected[i] && !immunity[i] && agent_node[i] == node;
    }

    // Get the indices of eligible agents using the boolean mask
    uint32_t* eligible_agents_indices = (uint32_t*)malloc(num_agents * sizeof(uint32_t));
    int count = 0;

    for (int i = 0; i < num_agents; ++i) {
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
        infected[selected_indices[i]] = 1;  // Assuming 1 represents True
        incubation_timer[selected_indices[i]] = 2; 
        infection_timer[selected_indices[i]] = rand() % (10) + 4; // Random integer between 4 and 14;
    }

    // Free allocated memory
    free(eligible_agents_indices);
    free(selected_indices);
}

void migrate( int num_agents, bool * infected, uint32_t * node ) {
    int fraction = (int)(0.01f*100);
    unsigned int counter = 0;
    for (int i = 0; i < num_agents; ++i) {
	if( infected[ i ] && rand()%100 < fraction )
	{
	    if( node[ i ] > 0 )
	    {
		node[ i ] --;
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
	uint32_t node_id = node[i];
	if( infected[ i ] ) {
	    infection_count[ node_id ]++;
	} else if( immunity[ i ] ) {
	    recovered_count[ node_id ]++;
	} else {
	    susceptible_count[ node_id ]++;
	}
    }
}
