#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <variant>

#include "matrix.h"
#include "vector.h"
#include "utils.h"
#include "bench.h"

#include "vector.cuh"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    int fSize = 100000000, fBlockSize = 1024, fGridSize = 32768;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCUDA = false,
        fcuBLAS = false,
        fFloat = false,
        fDouble = false,
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
    app.add_flag("--float", fFloat, "use single precision");
    app.add_flag("--double", fDouble, "use double precision");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCUDA = true;
        fcuBLAS = true;
    }

    using Number = std::variant<float, double>;
    std::vector<std::string> typeNames= {"float", "double"};

    Number sample = 0.0;
    if (fFloat) {
        sample = 0.0f;
    }
    auto typeName = typeNames[sample.index()];

    return std::visit([&](auto sample) {
        using T = decltype(sample);

        auto dataX = Utils::create_array<T>(fSize, 1.0);
        auto dataY = Utils::create_array<T>(fSize, 1.0);

        Utils::randomize_array<T>(dataX, fSize, fMin, fMax, fSeed);
        Utils::randomize_array<T>(dataY, fSize, fMin, fMax, fSeed);

        auto vX = Vector(dataX, fSize);
        auto vY = Vector(dataY, fSize);

        try {
            json jsonResult = {
                {"type",  typeName},
                {"size", fSize},
                {"seed", fSeed},
                {"range", {fMin,fMax}},
                {"cpu", Utils::cpuName()},
                {"gpu", CUDA::getDeviceName()},
                {"block_size", fBlockSize},
                {"grid_size", fGridSize},
                {"tests", json::array()},
                {"optimization", Utils::getOptimizationFlag()},
                {"tests", json::array()},
            };

            if (fCUDA) {
                auto runtime = "CUDA";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto cudaVx = VectorCUDA<T>(vX, fBlockSize, fGridSize);
                auto cudaVy = VectorCUDA<T>(vY, fBlockSize, fGridSize);

                auto bench = DotFlops(cudaVx, cudaVy);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fcuBLAS) {
                auto runtime = "cuBLAS";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto cudaVx = VectorCuBLAS<T>(vX);
                auto cudaVy = VectorCuBLAS<T>(vY);

                auto bench = DotFlops(cudaVx, cudaVy);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
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
    }, sample);
}