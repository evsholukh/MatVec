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

    int fSize = 100000000,
        fBlockSize = 1024,
        fGridSize = 32768,
        fSeed = std::chrono::system_clock::now().time_since_epoch().count();

    float fMin = -1.0,
          fMax = 1.0;

    bool fCPU = false,
        fOpenMP = false,
        fOpenBlas = false,
        fOpenCL = false,
        fClBlast = false;

    app.add_option("-n,--size", fSize, "vector size");
    app.add_option("-b,--block-size", fBlockSize, "block size");
    app.add_option("-g,--grid-size", fGridSize, "grid size");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openmp", fOpenMP, "OpenMP");
    app.add_flag("--openblas", fOpenBlas, "OpenBLAS");
    app.add_flag("--opencl", fOpenCL, "OpenCL");
    app.add_flag("--clblast", fClBlast, "clBLASt");

    CLI11_PARSE(app, argc, argv);

    std::cerr << "Creating vector.. " << fSize << std::endl;

    auto dataX = Utils::create_array<float>(fSize, 1.0);
    auto dataY = Utils::create_array<float>(fSize, 1.0);

    try {
        Utils::randomize_array<float>(dataX, fSize, fMin, fMax, fSeed);
        Utils::randomize_array<float>(dataY, fSize, fMin, fMax, fSeed);

        auto vX = Vector(dataX, fSize);
        auto vY = Vector(dataY, fSize);

        std::cerr << "Memory utilized: " << vX.size_mb() + vY.size_mb() << "MB" << std::endl;
        auto control = VectorCorrected(vX).dot(vY);

        json jsonResult = {
            {"size", fSize},
            {"control", control},
            {"seed", fSeed},
            {"min", fMin},
            {"max", fMax},
            {"cpu", Utils::cpuName().c_str()},
            {"gpu", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()},
            {"block_size", fBlockSize},
            {"grid_size", fGridSize},
            {"tests", json::array()},
        };

        auto result = 0.0f;
        if (fCPU) {
            auto duration = Utils::measure([&vX, &vY, &result]() { result = vX.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result}, 
                {"runtime", "C++"}, 
            });
        }
        if (fOpenMP) {
            auto ompVx = VectorOpenMP(vX);
            auto duration = Utils::measure([&ompVx, &vY, &result]() { result = ompVx.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", "OpenMP"},
            });
        }
        if (fOpenBlas) {
            auto bVx = VectorBLAS(vX);
            auto duration = Utils::measure([&bVx, &vY, &result]() { result = bVx.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", "OpenBLAS"},
            });
        }
        if (fOpenCL) {
            auto clVx = VectorOpenCL(vX, fBlockSize, fGridSize);
            auto clVy = VectorOpenCL(vY, fBlockSize, fGridSize);
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", "OpenCL"},
            });
        }
        if (fClBlast) {
            auto clVx = VectorCLBlast(vX);
            auto clVy = VectorCLBlast(vY);
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });

            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", "clBLASt"},
            });
        }
        std::cout << jsonResult.dump(4);

        delete[] dataX;
        delete[] dataY;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}