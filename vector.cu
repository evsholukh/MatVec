#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "cuda.cuh"
#include "CLI11.hpp"


int main(int argc, char **argv) {

    CLI::App app{"vector"};
    std::string size_str, blockSize_str, gridSize_str;
    
    app.add_option("-n,--size", size_str, "vector size");
    app.add_option("-b,--block-size", blockSize_str, "block size");
    app.add_option("-g,--grid-size", gridSize_str, "grid size");

    CLI11_PARSE(app, argc, argv);

    size_t size = 100000000, gridSize = 1024, blockSize = 1024;

    if (!size_str.empty()) {
        size = std::atoi(size_str.c_str());
    }
    if (!blockSize_str.empty()) {
        blockSize = std::atoi(blockSize_str.c_str());
    }
    if (!gridSize_str.empty()) {
        gridSize = std::atoi(gridSize_str.c_str());
    }
    try {
        std::cerr << "Creating vector (N: " << size << ")" << std::endl;

        auto data_x = Utils::create_array<float>(size, 1, 0.0001f);
        auto data_y = Utils::create_array<float>(size, 1, 0.0001f);

        Utils::randomize_array(data_x, size);
        Utils::randomize_array(data_y, size);

        auto vx = Vector<float>(data_x, size);
        auto vy = Vector<float>(data_y, size);

        std::cerr << "Memory utilized: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        printf("[");
        for (size_t i = 64; i <= blockSize; i *= 2) {
            for (size_t j = 1024; j <= gridSize; j *= 2) {
                {
                    auto cuda_rvx = VectorReduceCuda(vx, i, j);
                    auto cuda_rvy = VectorReduceCuda(vy, i, j);
                    auto value = 0.0f;
                    auto duration = Utils::measure([&cuda_rvx, &cuda_rvy, &value]() { value = cuda_rvx.dot(cuda_rvy); });
                    printf("{\"duration\": %f,"
                            "\"size\": %zd,"
                            "\"value\": %f,"
                            "\"block_size\": %zd,"
                            "\"grid_size\": %zd,"
                            "\"runtime\": \"%s\","
                            "\"device\": \"%s\"},\n",
                        duration, size, value, i, j, "CUDA Reduction", CUDA::deviceName().c_str());
                }
            }
        }
        {
            auto cuda_vx = VectorCuda(vx);
            auto cuda_vy = VectorCuda(vy);
            auto value = 0.0f;
            auto duration = Utils::measure([&cuda_vx, &cuda_vy, &value]() { value = cuda_vx.dot(cuda_vy); });
            printf("{\"duration\": %f,"
                    "\"size\": %zd,"
                    "\"value\": %f,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}",
                duration, size, value, "cuBLAS", CUDA::deviceName().c_str());
        }
        printf("]");

        delete[] data_x;
        delete[] data_y;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}