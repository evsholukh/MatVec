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
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{"vector"};

    int fSize = 1000000,
        fBlockSize = 1024,
        fGridSize = 32768,
        fSeed = -1;

    float fRandMin = -1.0,
          fRandMax = 1.0;

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

    app.add_flag("--cpu", fCPU, "CPU version");
    app.add_flag("--openmp", fOpenMP, "OpenMP verision");
    app.add_flag("--openblas", fOpenBlas, "OpenBLAS version");
    app.add_flag("--opencl", fOpenCL, "OpenCL version");
    app.add_flag("--blast", fClBlast, "clBLASt version");

    CLI11_PARSE(app, argc, argv);

    std::cerr << "Creating vector.. " << fSize << std::endl;

    auto dataX = Utils::create_array<float>(fSize, 1.0);
    auto dataY = Utils::create_array<float>(fSize, 1.0);

    try {
        Utils::randomize_array<float>(dataX, fSize, fRandMin, fRandMax, fSeed);
        Utils::randomize_array<float>(dataY, fSize, fRandMin, fRandMax, fSeed);

        auto vX = Vector(dataX, fSize);
        auto vY = Vector(dataY, fSize);

        std::cerr << "Memory utilized: " << vX.size_mb() + vY.size_mb() << "MB" << std::endl;

        if (fCPU) {
            auto result = 0.0f;
            auto duration = Utils::measure([&vX, &vY, &result]() { result = vX.dot(vY); });
            auto control = VectorCorrected(vX).dot(vY);

            json data = {
                {"duration", duration},
                {"value", result}, 
                {"control", control},
                {"size", fSize}, 
                {"runtime", "C++"},
                {"device", Utils::cpuName().c_str()},
            };
            std::cout << data.dump(4);
        }

        if (fOpenMP) {
            auto ompVx = VectorOpenMP(vX);
            auto result = 0.0f;
            auto duration = Utils::measure([&ompVx, &vY, &result]() { result = ompVx.dot(vY); });
            auto control = VectorCorrected(vX).dot(vY);

            json data = {
                {"duration", duration},
                {"result", result},
                {"control", control},
                {"size", fSize},
                {"runtime", "OpenMP"},
                {"device", Utils::cpuName().c_str()}, 
            };
            std::cout << data.dump(4);
        }
        if (fOpenBlas) {
            auto bVx = VectorBLAS(vX);
            auto result = 0.0f;
            auto duration = Utils::measure([&bVx, &vY, &result]() { result = bVx.dot(vY); });
            auto control = VectorCorrected(vX).dot(vY);

            json data = {
                {"duration", duration},
                {"result", result},
                {"control", control},
                {"size",  fSize},
                {"runtime", "OpenBLAS"},
                {"device", Utils::cpuName().c_str()},
            };
            std::cout << data.dump(4);
        }
        if (fOpenCL) {
            auto clVx = VectorOpenCL(vX, fBlockSize, fGridSize);
            auto clVy = VectorOpenCL(vY, fBlockSize, fGridSize);
            auto result = 0.0f;
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });
            auto control = VectorCorrected(vX).dot(vY);

            json data = {
                {"duration", duration},
                {"result", result},
                {"control", control},
                {"block_size", fBlockSize},
                {"grid_size", fGridSize},
                {"size", fSize},
                {"runtime", "OpenCL Reduction"},
                {"device", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()},
            };
            std::cout << data.dump(4);
        }
        if (fClBlast) {
            auto clVx = VectorCLBlast(vX);
            auto clVy = VectorCLBlast(vY);

            auto result = 0.0f;
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });
            auto control = VectorCorrected(vX).dot(vY);

            json data = {
                {"duration", duration},
                {"result", result},
                {"control", control},
                {"size", fSize},
                {"runtime", "clBLASt"},
                {"device", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()}
            };
            std::cout << data.dump(4);
        }
        delete[] dataX;
        delete[] dataY;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}