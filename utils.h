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


template<typename T>
T* random_vector(const size_t size) {
    T *data = new T[size];

    std::mt19937 generator(42);
    std::uniform_real_distribution<T> dist(-1, 1);

    for (size_t i = 0; i < size; i++) {
        data[i] = 1.0f; // dist(generator);
    }
    return data;
}
