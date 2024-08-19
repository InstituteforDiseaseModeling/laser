#include <stdint.h>
#include <omp.h>

void update_susceptibility_based_on_ri_timer(uint32_t count, uint16_t *ri_timer, uint8_t *susceptibility) { //,
                                             //float *age, int64_t tick) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < count; i++) {
        if (ri_timer[i] > 0) {
            ri_timer[i]--;
            if (ri_timer[i] == 0) {
                susceptibility[i] = 0;
            }
        }
    }
}

