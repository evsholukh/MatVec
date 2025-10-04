#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "opencl.h"
#include "cuda.cuh"

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

    bool fCUDA = false,
        fcuBLAS = false,
        fAll = false;

    app.add_option("-n,--size", fSize, "vector size");
    app.add_option("-b,--block-size", fBlockSize, "block size");
    app.add_option("-g,--grid-size", fGridSize, "grid size");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "lower value");
    app.add_option("--high", fMax, "higher value");

    app.add_flag("--cuda", fCUDA, "CUDA");
    app.add_flag("--cublas", fcuBLAS, "cuBLAS");

    app.add_flag("-a,--all", fAll, "All");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCUDA = true;
        fcuBLAS = true;
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
        std::cerr << "Running.." << std::endl;
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
            {"cpu", Utils::cpuName().c_str()},
            {"gpu", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()},
            {"o3", fO3},
            {"block_size", fBlockSize},
            {"grid_size", fGridSize},
            {"tests", json::array()},
        };

        if (fCUDA) {
            auto runtime = "CUDA";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto cudaVx = VectorReduceCuda(vX, fBlockSize, fGridSize);
            auto cudaVy = VectorReduceCuda(vY, fBlockSize, fGridSize);

            auto duration = Utils::measure([&vX, &vY, &result]() { result = vX.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result}, 
                {"runtime", runtime}, 
            });
        }
        if (fcuBLAS) {
            auto runtime = "cuBLAS";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto cudaVx = VectorCuda(vX);
            auto cudaVy = VectorCuda(vY);

            auto duration = Utils::measure([&vX, &vY, &result]() { result = vX.dot(vY); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", result}, 
                {"runtime", runtime}, 
            });
        }
        std::cout << jsonResult.dump(4);

        delete[] dataX;
        delete[] dataY;

        std::cerr << "Finished" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}