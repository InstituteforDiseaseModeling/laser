#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <unordered_map>
#include <vector>

extern "C" {

void tx_inner(
    uint8_t *susceptibilities,
    uint16_t *nodeids,
    float *forces,
    uint8_t *etimers,
    uint32_t count,
    float exp_mean,
    float exp_std,
    uint32_t *incidence,
    uint32_t num_nodes  // number of unique nodes
) {
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + time(NULL);  // Thread-local seed
        //uint32_t *local_incidence = calloc(num_nodes, sizeof(uint32_t));  // Thread-local storage for incidence

        #pragma omp for
        for (uint32_t i = 0; i < count; i++) {
            uint8_t susceptibility = susceptibilities[i];
            if (susceptibility > 0) {
                uint16_t nodeid = nodeids[i];
                float force = susceptibility * forces[nodeid]; // force of infection attenuated by personal susceptibility
                if (force > 0 && ((float)rand_r(&seed) / (float)RAND_MAX) < force) {  // thread-safe RNG
                    susceptibilities[i] = 0;  // set susceptibility to 0
                    // set exposure timer for newly infected individuals to a draw from a normal distribution, must be at least 1 day
                    float normal_draw = exp_mean + exp_std * ((float)rand_r(&seed) / RAND_MAX);
                    etimers[i] = fmax(1, round(normal_draw));
                    //local_incidence[nodeid] += 1;  // Increment thread-local incidence count
                }
            }
        }
#if 0
        // Aggregate thread-local incidence counts into the global incidence array
        #pragma omp critical
        {
            for (uint32_t j = 0; j < num_nodes; j++) {
                incidence[j] += local_incidence[j];
            }
        }

        free(local_incidence);  // Free the thread-local storage
#endif
    }
}


void tx_inner_atomic(
    uint8_t *susceptibilities,
    uint16_t *nodeids,
    float *forces,
    uint8_t *etimers,
    uint32_t count,
    float exp_mean,
    float exp_std,
    uint32_t *incidence
) {
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + time(NULL);  // Thread-local seed
        #pragma omp for
        for (uint32_t i = 0; i < count; i++) {
            uint8_t susceptibility = susceptibilities[i];
            if (susceptibility > 0) {
                uint16_t nodeid = nodeids[i];
                float force = susceptibility * forces[nodeid]; // force of infection attenuated by personal susceptibility
                if (force > 0 && ((float)rand_r(&seed) / (float)RAND_MAX) < force) {  // thread-safe RNG
                    susceptibilities[i] = 0;  // set susceptibility to 0
                                              // set exposure timer for newly infected individuals to a draw from a normal distribution, must be at least 1 day
                    float normal_draw = exp_mean + exp_std * ((float)rand_r(&seed) / RAND_MAX);
                    etimers[i] = fmax(1, round(normal_draw));
                    #pragma omp atomic
                    incidence[nodeid] += 1;
                }
            }
        }
    }
}

#if 0
    for (uint32_t i = 0; i < count; i++) {
        uint8_t susceptibility = susceptibilities[i];
        if (susceptibility > 0) {
            uint16_t nodeid = nodeids[i];
            float force = susceptibility * forces[nodeid]; // force of infection attenuated by personal susceptibility
            if (force > 0 && ((float)rand() / (float)RAND_MAX) < force) {  // draw random number < force means infection
                susceptibilities[i] = 0;  // set susceptibility to 0
                // set exposure timer for newly infected individuals to a draw from a normal distribution, must be at least 1 day
                float normal_draw = exp_mean + exp_std * ((float)rand() / RAND_MAX);
                etimers[i] = fmax(1, round(normal_draw));
                //#pragma omp atomic
                //incidence[nodeid] += 1;
            }
        }
    }
#endif

void tx_inner_nodes_v1(
    uint32_t count,
    unsigned int num_nodes,
    uint16_t * agent_node,
    uint8_t  * susceptibility,
    uint8_t  * incubation_timer,
    uint8_t  * infection_timer,
    uint16_t * new_infections_array,
    //int * num_eligible_agents_array,
    float incubation_period_constant
) {
    std::unordered_map<int, std::vector<int>> node2sus;

    // TBD: This first part should be done in the update_age census function
#pragma omp parallel
    {
        // Thread-local buffers to collect susceptible indices by node
        std::unordered_map<int, std::vector<int>> local_node2sus;

#pragma omp for nowait
        for (unsigned long int i = 0; i <= count; ++i) {
            //if (infection_timer[i]==0 && incubation_timer[i]==0 && susceptibility[i]==1) {
            if (susceptibility[i]==1) {
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
        //unsigned int num_eligible_agents = num_eligible_agents_array[node];
        //printf( "Node=%d, new_infections=%d, num_eligible_agents=%d\n", node, new_infections, num_eligible_agents );

        //if (new_infections > 0 && num_eligible_agents > 0 && node2sus.find(node) != node2sus.end()) {
        if (new_infections > 0 && node2sus.find(node) != node2sus.end()) {
            std::vector<int> &susceptible_indices = node2sus[node];
            int num_susceptible = susceptible_indices.size();
            int step = (new_infections >= num_susceptible) ? 1 : num_susceptible / new_infections;

            for (int i = 0, selected_count = 0; i < num_susceptible && selected_count < new_infections; i += step) {
                unsigned long int selected_id = susceptible_indices[i];
                //printf( "Infecting %ld\n", selected_id );
                incubation_timer[selected_id] = incubation_period_constant;
                susceptibility[selected_id] = 0;
                selected_count++;
            }
        }
    }
}

//static std::vector<std::vector<int>> local_node2sus(num_nodes);
//This is clearly a nasty hardcoding. TBD, don't do this.
//We are making this static so the container can be written in report and read in tx_inner_nodes.
static std::vector<std::vector<int>> local_node2sus(419);

// This function now assumes that report has been called first. But there is no check for that yet.
// report is what populates local_node2sus so we don't have to do a second census of susceptibles.
void tx_inner_nodes(
    uint32_t count,
    unsigned int num_nodes,
    uint8_t  * susceptibility,
    uint8_t  * incubation_timer,
    uint16_t * new_infections_array,
    float incubation_period_constant,
    uint32_t * infected_ids // Output: an array of arrays for storing infected IDs
) {
    uint32_t offsets[num_nodes];   // To store starting index for each node

    // Calculate offsets
    offsets[0] = 0;
    for (unsigned int node = 1; node < num_nodes; ++node) {
        offsets[node] = offsets[node - 1] + new_infections_array[node - 1];
    }

    // Second pass: Infect individuals by node in parallel
#pragma omp parallel for schedule(dynamic)
    for (unsigned int node = 0; node < num_nodes; ++node) {
        unsigned int new_infections = new_infections_array[node];
        //printf( "Finding %d new infections in node %d\n", new_infections, node );

        if (new_infections > 0) {
            std::vector<int> &susceptible_indices = local_node2sus[node];
            int num_susceptible = susceptible_indices.size();
            /*if( num_susceptible == 0 ) {
                printf( "WARNING: 0 susceptibles in node!\n" );
            }*/
            int step = (new_infections >= num_susceptible) ? 1 : num_susceptible / new_infections;

            // Get the starting index for this node's infections
            unsigned int start_index = offsets[node];

            for (int i = 0, selected_count = 0; i < num_susceptible && selected_count < new_infections; i += step) {
                unsigned long int selected_id = susceptible_indices[i];
                incubation_timer[selected_id] = incubation_period_constant;
                susceptibility[selected_id] = 0;
                selected_count++;
                // Write the infected ID into the pre-allocated array
                //printf( "Writing new infected id to index %d for node %d.\n", start_index + selected_count, node );
                infected_ids[start_index + selected_count] = selected_id;
            }
        }
    }
}

void report(
    unsigned long int count,
    int num_nodes,
    int32_t *age,
    uint16_t *node,
    unsigned char *infectious_timer, // max 255
    unsigned char *incubation_timer, // max 255
    bool *susceptibility, // yes or no
    unsigned char *susceptibility_timer, // max 255
    int *dod, // sim day
    uint32_t *susceptible_count,
    uint32_t *incubating_count,
    uint32_t *infectious_count,
    uint32_t *waning_count,
    uint32_t *recovered_count,
    unsigned int delta,
    unsigned int tick
) {
    uint32_t shard_index = tick % delta;

    //printf( "%s: count=%ld, num_nodes=%d", __FUNCTION__, count, num_nodes );
    #pragma omp parallel
    {
        std::vector<std::vector<int>> thread_local_node2sus(num_nodes);
        for (unsigned int node = 0; node < num_nodes; ++node) {
            local_node2sus[node].clear(); // Clear before inserting new data
        }

        // Thread-local buffers
        int *local_infectious_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_incubating_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_recovered_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_susceptible_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_waning_count = (int*) calloc(num_nodes, sizeof(int));

        #pragma omp for
        //for (size_t i = 0; i <= count; i++) {
        for (uint32_t i = shard_index; i < count; i += delta) {
            // Collect report 
            if (dod[i]>0) {
                int node_id = node[i];
                //printf( "Found live person at node %d: etimer=%d, itimer=%d, sus=%d.\n", node_id, incubation_timer[i], infectious_timer[i], susceptibility[i] );
                if (incubation_timer[i] > 0) {
                    //printf( "Found E in node %d.\n", node_id );
                    local_incubating_count[node_id]++;
                } else if (infectious_timer[i] > 0) {
                    //printf( "Found I in node %d.\n", node_id );
                    local_infectious_count[node_id]++;
                } else if (susceptibility[i]==0) {
                    //printf( "Found R in node %d.\n", node_id );
                    if (susceptibility_timer[i]>0) {
                        local_waning_count[node_id]++;
                    } else {
                        //printf( "ERROR? recording %lu as recovered: susceptibility_timer = %d.\n", i, susceptibility_timer[i] );
                        local_recovered_count[node_id]++;
                    }
                } else {
                    //printf( "Found S in node %d.\n", node_id );
                    local_susceptible_count[node_id]++;
                    thread_local_node2sus[node_id].push_back(i);
                }
            }
        }

        // Combine thread-local results
#pragma omp critical
        {
            for (unsigned int node = 0; node < num_nodes; ++node) {
                local_node2sus[node].insert(local_node2sus[node].end(),
                                            thread_local_node2sus[node].begin(),
                                            thread_local_node2sus[node].end());
            }
        }

        // Combine local counts into global counts
#pragma omp critical
        {
            for (int j = 0; j < num_nodes; ++j) {
                susceptible_count[j] += local_susceptible_count[j]; 
                incubating_count[j] += local_incubating_count[j];
                infectious_count[j] += local_infectious_count[j];
                waning_count[j] += local_waning_count[j];
                recovered_count[j] += local_recovered_count[j];
            }
        }

        // Free local buffers
        free(local_susceptible_count);
        free(local_incubating_count);
        free(local_infectious_count);
        free(local_waning_count);
        free(local_recovered_count);
    }
}
} // extern C (for C++)
