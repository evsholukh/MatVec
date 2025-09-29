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
#include "openmp.h"

#include "CLI11.hpp"


int main(int argc, char **argv) {

    CLI::App app{"vector"};

    int fSize = 1000000, fBlockSize = 1024, fGridSize = 32768, fSeed = 42;
    float fRandMin = -1.0, fRandMax = 1.0;
    bool fCPU = false,
        fOpenMP = false,
        fOpenBlas = false,
        fOpenCL = false,
        fClBlast = false;

    app.add_option("-n,--size", fSize, "vector size");
    app.add_option("-b,--block-size", fBlockSize, "block size");
    app.add_option("-g,--grid-size", fGridSize, "grid size");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fRandMin, "random lower value");
    app.add_option("--high", fRandMax, "random higher value");

    app.add_option("--cpu", fCPU, "CPU version");
    app.add_option("--openmp", fOpenMP, "OpenMP verision");
    app.add_option("--openblas", fOpenBlas, "OpenBLAS version");
    app.add_option("--blast", fClBlast, "clBLASt version");

    CLI11_PARSE(app, argc, argv);

    try {
         std::cerr << "Creating vector.. " << fSize << std::endl;

        auto dataX = Utils::create_array<float>(fSize, 1.0);
        auto dataY = Utils::create_array<float>(fSize, 1.0);

        Utils::randomize_array<float>(dataX, fSize, fRandMin, fRandMax, fSeed);
        Utils::randomize_array<float>(dataY, fSize, fRandMin, fRandMax, fSeed);

        auto vx = VectorFloat(dataX, fSize);
        auto vy = VectorFloat(dataY, fSize);

        std::cerr << "Memory utilized: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        if (fOpenCL) {
            auto vrx = VectorOpenCL(vx, fBlockSize, fGridSize);
            auto vry = VectorOpenCL(vy, fBlockSize, fGridSize);
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
                    "\"device\": \"%s\"}", duration, value, fBlockSize, fGridSize, fSize, "OpenCL Reduction", deviceName.c_str());

        }
        if (fCPU) {
            auto value = 0.0f;
            auto duration = Utils::measure([&vx, &vy, &value]() { value = vx.dot(vy); });

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}", duration, value, fSize, "C++", Utils::cpuName().c_str());
        }
        if (fOpenMP) {
            auto mp_vx = VectorOpenMP(vx);
            auto value = 0.0f;
            auto duration = Utils::measure([&mp_vx, &vy, &value]() { value = mp_vx.dot(vy); });

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}", duration, value, fSize, "OpenMP", Utils::cpuName().c_str());
        }
        if (fOpenBlas) {
            auto vbx = VectorBLAS(vx);
            auto value = 0.0f;
            auto duration = Utils::measure([&vbx, &vy, &value]() { value = vbx.dot(vy); });
            auto device = OpenCL::defaultDevice();
            auto deviceName = OpenCL::deviceName(device);

            printf("{\"duration\": %f,"
                    "\"value\": %f,"
                    "\"size\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}", duration, value, fSize, "OpenBLAS", Utils::cpuName().c_str());
        }
        if (fClBlast) {
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
                    "\"device\": \"%s\"}", duration, value, fSize, "clBLASt", deviceName.c_str());
        }
        delete[] dataX;
        delete[] dataY;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}