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

    CLI::App app{"MatMul"};
    std::string s, g, b;
    size_t size, gridSize, blockSize;

    app.add_option("-s,--size", s, "vector size");
    app.add_option("-g,--grid-size", g, "grid size");
    app.add_option("-b,--block-size", b, "blocks num");

    CLI11_PARSE(app, argc, argv);

    if (s.empty()) {
        std::cerr << "Size: ";
        std::cin >> size;
    } else {
        size = std::atoi(s.c_str());
    }
    if (g.empty()) {
        std::cerr << "Blocks: ";
        std::cin >> gridSize;
    } else {
        gridSize = std::atoi(g.c_str());
    }
    if (b.empty()) {
        std::cerr << "Threads: ";
        std::cin >> blockSize;
    } else {
        blockSize = std::atoi(b.c_str());
    }

    try {
        std::cerr << "Creating array (size: " << size << ").." << std::endl;

        auto data_x = Utils::create_array<float>(size);
        auto data_y = Utils::create_array<float>(size);

        Utils::randomize_array(data_x, size);
        Utils::randomize_array(data_y, size);

        auto vx = Vector<float>(data_x, size);
        auto vy = Vector<float>(data_y, size);

        auto cuda_vx = VectorCuda(vx);
        auto cuda_vy = VectorCuda(vy);

        auto cuda_rvx = VectorReduceCuda(vx, blockSize, gridSize);
        auto cuda_rvy = VectorReduceCuda(vy, blockSize, gridSize);

        std::cerr << "Memory utilized: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        auto value = 0.0f;
        auto duration = 0.0f;
        printf("[");
        {
            duration = Utils::measure([&cuda_rvx, &cuda_rvy, &value]() { value = cuda_rvx.dot(cuda_rvy); });
            printf("{\"size\": %d, \"duration\": %f, \"value\": %f, \"block_size\": %d, \"grid_size\": %d, \"runtime\": \"%s\", \"device\": \"%s\"}\n",
                size, duration, value, blockSize, gridSize, "CUDA Reduction", "TODO");
        }
        printf("]");

        delete[] data_x;
        delete[] data_y;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}