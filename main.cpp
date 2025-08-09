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
    std::string n, bs;

    app.add_option("-n,--count", n, "blocks");
    app.add_option("-b,--block-size", bs, "block size");

    CLI11_PARSE(app, argc, argv);

    auto device = OpenCL::defaultDevice();
    auto deviceName = OpenCL::deviceName(device);

    size_t N = 1, blockSize = OpenCL::maxGroupSize(device);

    if (n.empty()) {
        std::cerr << "N: ";
        std::cin >> N;
    } else {
        N = std::atoi(n.c_str());
    }
    if (!bs.empty()) {
        blockSize = std::atoi(bs.c_str());
    }
    auto size = N * blockSize;
    try {
        std::cerr << "Creating array (size: " << size << ").." << std::endl;

        auto dataX = Utils::create_array<float>(size, blockSize, 0.0001f);
        auto dataY = Utils::create_array<float>(size, blockSize, 0.0001f);

        Utils::randomize_array(dataX, size);
        Utils::randomize_array(dataY, size);

        auto vx = Vector<float>(dataX, size);
        auto vy = Vector<float>(dataY, size);

        std::cerr << "Memory size: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        auto value = 0.0f;
        auto duration = 0.0f;
        printf("[");
        {
            duration = Utils::measure([&vx, &vy, &value]() { value = vx.dot(vy); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"count\": %d, \"runtime\": \"%s\", \"device\": \"%s\"},\n", duration, value, blockSize, N, "C++", deviceName.c_str());
        }
        {
            auto vbx = VectorBLAS(vx); 
            auto vby = VectorBLAS(vy);

            duration = Utils::measure([&vbx, &vby, &value]() { value = vbx.dot(vby); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"count\": %d, \"runtime\": \"%s\", \"device\": \"%s\"},\n", duration, value, blockSize, N, "OpenBLAS", deviceName.c_str());
        }
        {
            auto vrx = VectorOpenCL(vx, blockSize);
            auto vry = VectorOpenCL(vy, blockSize);
            duration = Utils::measure([&vrx, &vry, &value]() { value = vrx.dot(vry); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"count\": %d, \"runtime\": \"%s\", \"device\": \"%s\"},\n", duration, value, blockSize, N, "OpenCL Reduction", deviceName.c_str());
        }
        {
            auto cl_vx = VectorCLBlast(vx);
            auto cl_vy = VectorCLBlast(vy);
            duration = Utils::measure([&cl_vx, &cl_vy, &value]() { value = cl_vx.dot(cl_vy); });
            printf("{\"duration\": %f, \"value\": %f, \"block_size\": %d, \"count\": %d, \"runtime\": \"%s\", \"device\": \"%s\"}", duration, value, blockSize, N, "clBLASt", deviceName.c_str());
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