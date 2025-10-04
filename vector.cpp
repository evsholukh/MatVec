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

    int fSize = 100000000, fBlockSize = 1024, fGridSize = 32768;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCPU = false,
        fOpenMP = false,
        fOpenBLAS = false,
        fOpenCL = false,
        fClBlast = false,
        fAll = false;

    app.add_option("-n,--size", fSize, "vector size");
    app.add_option("-b,--block-size", fBlockSize, "block size");
    app.add_option("-g,--grid-size", fGridSize, "grid size");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openmp", fOpenMP, "OpenMP");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--opencl", fOpenCL, "OpenCL");
    app.add_flag("--clblast", fClBlast, "clBLASt");

    app.add_flag("-a,--all", fAll, "All");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCPU = true;
        fOpenMP = true;
        fOpenBLAS = true;
        fOpenCL = true;
        fClBlast = true;
    }
    std::cerr << "Creating array " << fSize << ".." << std::endl;
    auto dataX = Utils::create_array<float>(fSize, 1.0);
    Utils::randomize_array<float>(dataX, fSize, fMin, fMax, fSeed);
    auto vX = Vector(dataX, fSize);
    std::cerr << "Memory utilized: " << vX.size_mb() << "MB" << std::endl;

    std::cerr << "Creating array " << fSize << ".." << std::endl;
    auto dataY = Utils::create_array<float>(fSize, 1.0);
    Utils::randomize_array<float>(dataY, fSize, fMin, fMax, fSeed);
    auto vY = Vector(dataY, fSize);
    std::cerr << "Memory utilized: " << vY.size_mb() << "MB" << std::endl;

    try {
        std::cerr << "Running control.." << std::endl;
        auto result = VectorCorrected(vX).dot(vY);

        auto fO3 = false;
        #ifdef OPT_LEVEL_O3
            fO3 = true;
        #endif

        json jsonResult = {
            {"size", fSize},
            {"result", result},
            {"seed", fSeed},
            {"min", fMin},
            {"max", fMax},
            {"cpu", Utils::cpuName()},
            {"gpu", OpenCL::deviceName(OpenCL::defaultDevice())},
            {"o3", fO3},
            {"block_size", fBlockSize},
            {"grid_size", fGridSize},
            {"tests", json::array()},
            {"opencl",
                {
                    {"platform_name", OpenCL::platformName(OpenCL::defaultPlatform())},
                    {"device_version", OpenCL::deviceVersion(OpenCL::defaultDevice())},
                    {"driver_version", OpenCL::driverVersion(OpenCL::defaultDevice())},
                }
            }
        };

        if (fCPU) {
            auto runtime = "C++";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto duration = Utils::measure([&vX, &vY, &result]() { result = vX.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result}, 
                {"runtime", runtime}, 
            });
        }
        if (fOpenMP) {
            auto runtime = "OpenMP";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto ompVx = VectorOpenMP(vX);
            auto duration = Utils::measure([&ompVx, &vY, &result]() { result = ompVx.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", runtime},
            });
        }
        if (fOpenBLAS) {
            auto runtime = "OpenBLAS";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto bVx = VectorBLAS(vX);
            auto duration = Utils::measure([&bVx, &vY, &result]() { result = bVx.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", runtime},
            });
        }
        if (fOpenCL) {
            auto runtime = "OpenCL";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto clVx = VectorOpenCL(vX, fBlockSize, fGridSize);
            auto clVy = VectorOpenCL(vY, fBlockSize, fGridSize);
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", runtime},
            });
        }
        if (fClBlast) {
            auto runtime = "clBLASt";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto clVx = VectorCLBlast(vX);
            auto clVy = VectorCLBlast(vY);
            auto duration = Utils::measure([&clVx, &clVy, &result]() { result = clVx.dot(clVy); });

            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result},
                {"runtime", runtime},
            });
        }
        std::cout << jsonResult.dump(4) << std::endl;

        delete[] dataX;
        delete[] dataY;

        std::cerr << "Finished" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}