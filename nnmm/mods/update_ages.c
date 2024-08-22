// update_ages.c

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include <immintrin.h>

#define SIMD_WIDTH 8     // AVX2 processes 8 integers at a time
unsigned recovered_counter = 0;

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
 *
 * SQL: UPDATE ages SET age = age + 1.0f/365.0f WHERE age >= 0 
 * AND index >= start_idx AND index <= stop_idx;
 */
const float one_day = 1.0f/365.0f;
//void update_ages(unsigned long int stop_idx, float *ages) {
void update_ages(unsigned long int stop_idx, int *ages) {
    //printf( "%s: from %d to %d.\n", __FUNCTION__, start_idx, stop_idx );
    #pragma omp parallel for
    for (size_t i = 0; i <= stop_idx; i++) {
        if( ages[i] >= 0 )
        {
            //ages[i] += one_day;
            ages[i] ++;
        }
    }
}

// avxv2
void update_ages_simd(unsigned long int stop_idx, float *ages) {
    // Note that we could make sure not to increase ages of agents who have already exceeded
    // their expected lifespan, but it's probably more expensive to check the expected_lifespan
    // of each agent first, compare to age, than just unconditionally add to everyone.
    #pragma omp parallel for
    for (size_t i = 0; i <= stop_idx; i += 8) {
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

void update_ages_and_report(
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
    uint32_t *recovered_count 
) {
    //printf( "%s: count=%ld, num_nodes=%d", __FUNCTION__, count, num_nodes );
    #pragma omp parallel
    {
        // Thread-local buffers
        int *local_infectious_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_incubating_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_recovered_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_susceptible_count = (int*) calloc(num_nodes, sizeof(int));
        int *local_waning_count = (int*) calloc(num_nodes, sizeof(int));

        #pragma omp for
        for (size_t i = 0; i <= count; i++) {
            // Update ages
            if (age[i] >= 0) {
                age[i]++;
            }

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
                        local_recovered_count[node_id]++;
                    }
                } else {
                    //printf( "Found S in node %d.\n", node_id );
                    local_susceptible_count[node_id]++;
                }
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

