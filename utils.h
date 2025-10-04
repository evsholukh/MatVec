#pragma once

#include <chrono>
#include <random>

#include <array>
#include <cstring>
#include <algorithm>
#include <unordered_map>

#include <openblas_config.h> 

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

class Utils {

public:
    template <typename F>
    static float measure(F func) {
        auto start_time = std::chrono::high_resolution_clock::now();
        func();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;

        return elapsed.count() * 1000;
    }

    template <typename T>
    static T *create_array(const size_t size, const T val = T(0), const size_t align = 1) {
        const int count = (size + align - 1) / align;
        const int global_size = align * count;

        auto data = new T[global_size];

        fill_array(data, global_size, T(0));
        fill_array(data, size, val);

        return data;
    }

    template <typename T>
    static void randomize_array(
        T *data,
        const size_t size,
        const T low = T(-1.0),
        const T high = T(1.0),
        int seed = 42) {

        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> dist(low, high);

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

    static inline void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    }

    static inline void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

    static std::string cpuName() {
        char cpuName[49] = {0};
        unsigned int regs[4];

        for (int i = 0; i < 3; i++) {
            #ifdef _MSC_VER
            __cpuid((int*)regs, 0x80000002 + i);
            #else
            __cpuid(0x80000002 + i, regs[0], regs[1], regs[2], regs[3]);
            #endif
            std::memcpy(cpuName + i * 16, regs, sizeof(regs));
        }  
        
        std::string str(cpuName);
        rtrim(str);

        return str;
    }

    static std::string getOpenMPVersion() {
        #ifndef _OPENMP
            #define _OPENMP 0
        #endif
        std::unordered_map<unsigned,std::string> map{
            {199810,"1.0"},
            {200203,"2.0"},
            {200505,"2.5"},
            {200805,"3.0"},
            {201107,"3.1"},
            {201307,"4.0"},
            {201511,"4.5"},
            {201811,"5.0"},
            {202011,"5.1"},
            {202111,"5.2"},
            {202411,"6.0"},
            {0, ""}
        };
        return map.at(_OPENMP);
    }

    static std::string getStandardVersion() {
        std::unordered_map<unsigned,std::string> map {
            {201103L, "C++11"},
            {201402L, "C++14"},
            {201703L, "C++17"},
            {202002L, "C++20"},
        };
        return map.at(__cplusplus);
    }

    static std::string getOpenBLASVersion() {
        std::string str(OPENBLAS_VERSION);
        ltrim(str);
        rtrim(str);

        return str;
    }

    static std::string getCompilerVersion() {
        return __VERSION__;
    }

    static std::string getOptimizationFlag() {
        #ifndef OPT_LEVEL
            #define OPT_LEVEL "default"
        #endif
        return OPT_LEVEL;
    }
};
