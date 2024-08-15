#include <stdint.h>
#include <omp.h>

void update_susceptibility_based_on_sus_timer(uint32_t count, uint8_t* susceptibility_timer, uint8_t* susceptibility) {
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

