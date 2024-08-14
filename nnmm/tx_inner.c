#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

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
        uint32_t *local_incidence = calloc(num_nodes, sizeof(uint32_t));  // Thread-local storage for incidence

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
                    local_incidence[nodeid] += 1;  // Increment thread-local incidence count
                }
            }
        }

        // Aggregate thread-local incidence counts into the global incidence array
        #pragma omp critical
        {
            for (uint32_t j = 0; j < num_nodes; j++) {
                incidence[j] += local_incidence[j];
            }
        }

        free(local_incidence);  // Free the thread-local storage
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
