#include <stdint.h>
#include <stdio.h>
#include <omp.h>

extern "C" {

void update_susceptibility_based_on_sus_timer(uint32_t count, uint16_t* susceptibility_timer, uint8_t* susceptibility) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < count; i++) {
        if (susceptibility_timer[i] > 0) {
            susceptibility_timer[i]--;
            if (susceptibility_timer[i] == 0) {
                susceptibility[i] = 1;
            }
        }
    }
}

void update_susceptibility_timer_strided_shards(uint32_t count, uint16_t* susceptibility_timer, uint8_t* susceptibility, uint32_t delta, uint32_t tick) {
    // Calculate shard index based on current tick
    uint32_t shard_index = tick % delta;

    #pragma omp parallel for
    for (uint32_t i = shard_index; i < count; i += delta) {
        if (susceptibility_timer[i] > 0) {
            // Played around with a few different ways of decrementing values > 1 without slowing things down. This seems OK.
            susceptibility_timer[i] = (susceptibility_timer[i] > delta) ? (susceptibility_timer[i] - delta) : 0;

            if (susceptibility_timer[i] == 0) {
                susceptibility[i] = 1;
            }
        }
    }
}

}
