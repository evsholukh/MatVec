#include <iostream>

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
        std::cerr << "Creating vector.. (" << size << ")" << std::endl;

        auto dataX = Utils::create_array<float>(size, 1, 0.0001f);
        auto dataY = Utils::create_array<float>(size, 1, 0.0001f);

        Utils::randomize_array(dataX, size);
        Utils::randomize_array(dataY, size);

        auto vx = Vector<float>(dataX, size);
        auto vy = Vector<float>(dataY, size);

        std::cerr << "Memory utilized: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        printf("[");
        for (size_t i = 64; i <= blockSize; i *= 2) {
            for (size_t j = 1024; j <= gridSize; j *= 2) {
                {
                    auto vrx = VectorOpenCL(vx, i, j);
                    auto vry = VectorOpenCL(vy, i, j);
                    auto value = 0.0f;
                    auto duration = Utils::measure([&vrx, &vry, &value]() { value = vrx.dot(vry); });
                    auto device = OpenCL::defaultDevice();
                    auto deviceName = OpenCL::deviceName(device);

                    printf("{\"duration\": %f,"
                            "\"value\": %f,"
                            "\"block_size\": %d,"
                            "\"grid_size\": %d,"
                            "\"size\": %d,"
                            "\"runtime\": \"%s\","
                            "\"device\": \"%s\"},\n", duration, value, i, j, size, "OpenCL Reduction", deviceName.c_str());
                }
            }
        }
        {
            auto value = 0.0f;
            auto duration = Utils::measure([&vx, &vy, &value]() { value = vx.dot(vy); });

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"},\n", duration, value, size, "C++", "CPU");
        }
        {
            auto vbx = VectorBLAS(vx); 
            auto vby = VectorBLAS(vy);
            auto value = 0.0f;
            auto duration = Utils::measure([&vbx, &vby, &value]() { value = vbx.dot(vby); });
            auto device = OpenCL::defaultDevice();
            auto deviceName = OpenCL::deviceName(device);

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"},\n", duration, value, size, "OpenBLAS", "CPU");
        }
        {
            auto cl_vx = VectorCLBlast(vx);
            auto cl_vy = VectorCLBlast(vy);
            auto value = 0.0f;
            auto duration = Utils::measure([&cl_vx, &cl_vy, &value]() { value = cl_vx.dot(cl_vy); });
            auto device = OpenCL::defaultDevice();
            auto deviceName = OpenCL::deviceName(device);

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}", duration, value, size, "clBLASt", deviceName.c_str());
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