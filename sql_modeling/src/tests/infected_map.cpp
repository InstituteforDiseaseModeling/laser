#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib> // for rand
#include <ctime> // for time
#include <algorithm> // for std::count
#include <chrono> // for std::count

std::vector<int> selectNewInfecteds(std::vector<bool>& array) {
    std::vector<int> randomIndices;
    // Logic to select random indices
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] == false && std::rand() % 100 < 1) { // Select 10% of uninfected agents
            //printf( "Selected idx %d for infection.\n", i );
            randomIndices.push_back(i);
            array[i] = true;
        }
    }
 
    return randomIndices;
}

void test_map() {
    // Seed the random number generator
    std::srand(std::time(nullptr));

    // Initialize variables
    int timesteps = 7300;
    std::vector<bool> array(1000000, false); // Assuming 1000 agents
    std::unordered_map<int, std::vector<int>> infecteds;

    // Main loop
    for (int timestep = 0; timestep < timesteps; ++timestep) {
        // Get random indices
        std::vector<int> randomIndices = selectNewInfecteds(array);

        // Calculate timer values for each random index
        std::vector<int> timerValues(randomIndices.size());
        for (size_t i = 0; i < randomIndices.size(); ++i) {
            timerValues[i] = std::rand() % 12 + 5; // Random value between 5 and 16
        }

        // Calculate expiration times for each random index
        std::vector<int> expirationTimes(randomIndices.size());
        for (size_t i = 0; i < expirationTimes.size(); ++i) {
            expirationTimes[i] = timerValues[i] + timestep;
        }

        // Assign or extend the infecteds map with the corresponding indices for each expiration time
        for (size_t i = 0; i < expirationTimes.size(); ++i) {
            int expirationTime = expirationTimes[i];
            int index = randomIndices[i];
            infecteds[expirationTime].push_back(index);
        }

        // Check the priority queue to identify indices where the timer has expired
        if (infecteds.find(timestep) != infecteds.end()) {
            std::vector<int>& recovereds = infecteds[timestep];
            for (int idx : recovereds) {
                array[idx] = false;
            }
            infecteds.erase(timestep);
        }
        // Print current status
        //std::cout << "T=" << timestep << ", infected=" << std::count(array.begin(), array.end(), true) << "." << std::endl;
    }
}

void test_vector() {
    // Initialize variables
    int timesteps = 7300;
    std::vector<bool> array(1000000, false); // Assuming 1000 agents
    std::vector<int> infectionTimers(1000000, 0);

    // Seed the random number generator
    std::srand(std::time(nullptr));

    // Main loop
    for (int timestep = 0; timestep < timesteps; ++timestep) {
        // Infect a fraction of uninfecteds
        std::vector<int> randomIndices = selectNewInfecteds(array);
        std::vector<int> uninfectedIndices;
        for (int idx : randomIndices) {
            if (infectionTimers[idx] == 0) {
                uninfectedIndices.push_back(idx);
            }
        }
        // Initialize their timers
        for (int idx : uninfectedIndices) {
            infectionTimers[idx] = std::rand() % 12 + 5; // Random value between 5 and 16
        }

        // Decrement infection timers
        for (size_t i = 0; i < array.size(); ++i) {
            if (array[i]) {
                infectionTimers[i]--;
            }
        }

        // Flip bools to false where infection timers are 0
        for (size_t i = 0; i < array.size(); ++i) {
            if (infectionTimers[i] == 0) {
                array[i] = false;
            }
        }

        // Print current status
        //std::cout << "T=" << timestep << ", infected=" << std::count(array.begin(), array.end(), true) << "." << std::endl;
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    test_map();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Map took " << duration.count() << std::endl;

    test_vector();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Vector took " << duration.count() << std::endl;
}
