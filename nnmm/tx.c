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

void tx_inner_nodes(
    uint32_t count,
    unsigned int num_nodes,
    uint16_t * agent_node,
    uint8_t  * susceptibility,
    uint8_t  * incubation_timer,
    uint8_t  * infection_timer,
    uint16_t * new_infections_array,
    float incubation_period_constant,
    uint32_t * infected_ids // Output: an array of arrays for storing infected IDs
) {
    // Local maps for each thread
    std::vector<std::vector<int>> local_node2sus(num_nodes);

    uint32_t offsets[num_nodes];   // To store starting index for each node

    // Calculate offsets
    offsets[0] = 0;
    for (unsigned int node = 1; node < num_nodes; ++node) {
        offsets[node] = offsets[node - 1] + new_infections_array[node - 1];
    }

    // First pass: gather susceptible individuals by node in parallel
#pragma omp parallel
    {
        // Thread-local buffers to collect susceptible indices by node
        std::vector<std::vector<int>> thread_local_node2sus(num_nodes);

#pragma omp for nowait
        for (unsigned long int i = 0; i < count; ++i) {
            if (susceptibility[i] == 1) {
                int node = agent_node[i];
                thread_local_node2sus[node].push_back(i);
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
    }

    // Second pass: Infect individuals by node in parallel
#pragma omp parallel for schedule(dynamic)
    for (unsigned int node = 0; node < num_nodes; ++node) {
        unsigned int new_infections = new_infections_array[node];

        if (new_infections > 0) {
            std::vector<int> &susceptible_indices = local_node2sus[node];
            int num_susceptible = susceptible_indices.size();
            int step = (new_infections >= num_susceptible) ? 1 : num_susceptible / new_infections;

            // Get the starting index for this node's infections
            unsigned int start_index = offsets[node];

            for (int i = 0, selected_count = 0; i < num_susceptible && selected_count < new_infections; i += step) {
                unsigned long int selected_id = susceptible_indices[i];
                incubation_timer[selected_id] = incubation_period_constant;
                susceptibility[selected_id] = 0;
                selected_count++;
                // Write the infected ID into the pre-allocated array
                //printf( "Writing new infected id to index %d.\n", start_index + selected_count );
                infected_ids[start_index + selected_count] = selected_id;
            }
        }
    }
}


} // extern C (for C++)
