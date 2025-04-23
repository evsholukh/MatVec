#pragma once

#include <chrono>

class Utils {

public:

    template <typename F>
    static float measure(F func) {
        auto start_time = std::chrono::high_resolution_clock::now();
        func();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;
    
        return elapsed.count();
    }
};


