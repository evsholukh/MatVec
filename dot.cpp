#include <iostream>

#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <omp.h>
#include <openblas_config.h>
#include <variant>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "opencl.h"
#include "openblas.h"
#include "openmp.h"

#include "bench.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    int fSize = 100000000, fBlockSize = 1024, fGridSize = 32768;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCorrect = false,
        fCPU = false,
        fOpenMP = false,
        fOpenBLAS = false,
        fOpenCL = false,
        fClBlast = false,
        fFloat = false,
        fDouble = false,
        fAll = false;

    app.add_option("-n,--size", fSize, "vector size");
    app.add_option("-b,--block-size", fBlockSize, "block size");
    app.add_option("-g,--grid-size", fGridSize, "grid size");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--correct", fCorrect, "use correction");
    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openmp", fOpenMP, "OpenMP");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--opencl", fOpenCL, "OpenCL");
    app.add_flag("--clblast", fClBlast, "clBLASt");

    app.add_flag("-a,--all", fAll, "All");
    app.add_flag("--float", fFloat, "use single precision");
    app.add_flag("--double", fDouble, "use double precision");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCorrect = true;
        fCPU = true;
        fOpenMP = true;
        fOpenBLAS = true;
        fOpenCL = true;
        fClBlast = true;
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

        std::cerr << "Creating array " << fSize << ".." << std::endl;
        auto dataX = Utils::create_array<T>(fSize, 1.0);
        Utils::randomize_array<T>(dataX, fSize, fMin, fMax, fSeed);
        auto vX = Vector(dataX, fSize);
        std::cerr << "Memory utilized: " << vX.size_mb() << "MB" << std::endl;

        std::cerr << "Creating array " << fSize << ".." << std::endl;
        auto dataY = Utils::create_array<T>(fSize, 1.0);
        Utils::randomize_array<T>(dataY, fSize, fMin, fMax, fSeed);
        auto vY = Vector(dataY, fSize);
        std::cerr << "Memory utilized: " << vY.size_mb() << "MB" << std::endl;

        try {
            json jsonResult = {
                {"type",  typeName},
                {"size", fSize},
                {"seed", fSeed},
                {"range", {fMin,fMax}},
                {"cpu", Utils::cpuName()},
                {"gpu", OpenCL::getDeviceName(OpenCL::defaultDevice())},
                {"block_size", fBlockSize},
                {"grid_size", fGridSize},
                {"tests", json::array()},
                {"optimization", Utils::getOptimizationFlag()},
                {"env", {
                    {
                        {"runtime", "C++"},
                        {"version", OpenCL::getCompilerVersion()},
                    },
                    {
                        {"runtime", "OpenMP"},
                        {"version", VectorOpenMP<>::getOpenMPVersion()},
                    },
                    {
                        {"runtime", "OpenBLAS"},
                        {"version", VectorBLAS<>::getOpenBLASVersion()},
                    },
                    {
                        {"runtime", "OpenCL"},
                        {"version", OpenCL::getDeviceVersion(OpenCL::defaultDevice())},
                    },
                    {
                        {"runtime", "CLBlast"},
                        {"version", VectorCLBlast<>::getCLBlastVersion()},
                    },
                }},
            };
            T result = T(0);

            if (fCorrect) {
                auto runtime = "C++*";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = VectorCorrected(vX);
                auto bench = DotFlops(x, vY);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
                    {"result", result},
                    {"runtime", runtime},
                });
            }
            if (fCPU) {
                auto runtime = "C++";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto bench = DotFlops(vX, vY);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
                    {"result", result},
                    {"runtime", runtime}, 
                });
            }
            if (fOpenMP) {
                auto runtime = "OpenMP";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto ompVx = VectorOpenMP(vX);
                auto bench = DotFlops(ompVx, vY);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
                    {"result", result},
                    {"runtime", runtime},
                });
            }
            if (fOpenBLAS) {
                auto runtime = "OpenBLAS";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto bVx = VectorBLAS(vX);
                auto bench = DotFlops(bVx, vY);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
                    {"result", result},
                    {"runtime", runtime},
                });
            }
            if (fOpenCL) {
                auto runtime = "OpenCL";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto clVx = VectorOpenCL(vX, fBlockSize, fGridSize);
                auto clVy = VectorOpenCL(vY, fBlockSize, fGridSize);

                auto bench = DotFlops(clVx, clVy);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
                    {"result", result},
                    {"runtime", runtime},
                });
            }
            if (fClBlast) {
                auto runtime = "CLBlast";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto clVx = VectorCLBlast(vX);
                auto clVy = VectorCLBlast(vY);

                auto bench = DotFlops(clVx, clVy);
                auto metric = bench.perform();
                auto flops = metric.gflops();
                auto result = metric.result();

                jsonResult["tests"].push_back({
                    {"gflops", flops},
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
    }, sample);
}