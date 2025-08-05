#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "opencl.h"
#include "openblas.h"

#include "CLI11.hpp"


int main(int argc, char **argv) {

    CLI::App app{"MatMul"};
    std::string n;

    app.add_option("-n,--size", n, "size");

    CLI11_PARSE(app, argc, argv);
    size_t N, blockSize = 1024;

    if (n.empty()) {
        std::cerr << "N: ";
        std::cin >> N;
    } else {
        N = std::atoi(n.c_str());
    }
    size_t blocks = N / blockSize;

    try {
        std::cerr << "Creating array (size: " << N << ").." << std::endl;

        auto dataX = Utils::create_array<float>(N, blockSize, 0.0001f);
        auto dataY = Utils::create_array<float>(N, blockSize, 0.0001f);

        Utils::randomize_array(dataX, N);
        Utils::randomize_array(dataY, N);

        auto vx = Vector<float>(dataX, N);
        auto vy = Vector<float>(dataY, N);

        std::cerr << "Memory size: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        auto value = 0.0f;
        auto duration = 0.0f;
        printf("[");
        {
            duration = Utils::measure([&vx, &vy, &value]() { value = vx.dot(vy); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"blocks\": %d, \"runtime\": \"%s\"},\n", duration, value, N, 1, "C++");
        }
        {
            auto vbx = VectorBLAS(vx); 
            auto vby = VectorBLAS(vy);

            duration = Utils::measure([&vbx, &vby, &value]() { value = vbx.dot(vby); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"blocks\": %d, \"runtime\": \"%s\"},\n", duration, value, N, 1, "OpenBLAS");
        }
        {
            for (size_t i = 2; i <= 1024; i *= 2) {
                auto vrx = VectorOpenCL(vx, i);
                auto vry = VectorOpenCL(vy, i);
                duration = Utils::measure([&vrx, &vry, &value]() { value = vrx.dot(vry); });
                printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"blocks\": %d, \"runtime\": \"%s\"},\n", duration, value, i, N / i, "OpenCL Reduction");
            }
        }
        {
            auto cl_vx = VectorCLBlast(vx);
            auto cl_vy = VectorCLBlast(vy);
            duration = Utils::measure([&cl_vx, &cl_vy, &value]() { value = cl_vx.dot(cl_vy); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"blocks\": %d, \"runtime\": \"%s\"}", duration, value, N, 1, "clBLASt");
        }
        printf("]");

        delete[] dataX;
        delete[] dataY;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}