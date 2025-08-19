#pragma once

#include <chrono>
#include <random>

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

    template <typename T>
    static T *create_array(const size_t size, const size_t align = 1, const T val = T(0)) {
        const int count = (size + align - 1) / align;
        const int global_size = align * count;

        auto data = new T[global_size];

        fill_array(data, global_size, T(0));
        fill_array(data, size, val);

        return data;
    }

    template <typename T>
    static void randomize_array(T *data, const size_t size) {
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> dist(-1, 1);

        for (size_t i = 0; i < size; i++) {
            data[i] *= dist(generator);
        }
    }

    template <typename T>
    static void fill_array(T *data, const size_t size, const T val = T(0)) {
        for (size_t i = 0; i < size; i++) {
            data[i] = val;
        }
    }
};
