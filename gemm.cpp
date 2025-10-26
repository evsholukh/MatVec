
#include <iostream>
#include <variant>

#include "utils.h"
#include "matrix.h"
#include "openblas.h"
#include "opencl.h"
#include "openmp.h"

#include "bench.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    int fM = 1000, fN = 1000, fK = 1000;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCPU = false,
         fOpenMP = false,
         fOpenBLAS = false,
         fClBlast = false,
         fOpenCL = false,
         fFloat = false,
         fDouble = false, 
         fAll = false;

    app.add_option("-m", fM, "x-rows");
    app.add_option("-n", fN, "y-cols");
    app.add_option("-k", fK, "x-cols, y-rows");

    app.add_option("-s,--seed", fSeed, "random seed");

    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openmp", fOpenMP, "OpenMP");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--opencl", fOpenCL, "OpenCL");
    app.add_flag("--clblast", fClBlast, "CLBlast");

    app.add_flag("-a,--all", fAll, "All");

    app.add_flag("--float", fFloat, "single precision");
    app.add_flag("--double", fDouble, "double precision");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
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

        auto arrX = Utils::create_array<T>(fM*fK, 1.0);
        auto arrY = Utils::create_array<T>(fK*fN, 1.0);
        auto arrZ = Utils::create_array<T>(fK*fK);

        Utils::randomize_array<T>(arrX, fM*fK, fMin, fMax, fSeed);
        Utils::randomize_array<T>(arrY, fK*fN, fMin, fMax, fSeed);

        auto matY = Matrix(arrY, fK, fN);
        auto matX = Matrix(arrX, fM, fK);
        auto matZ = Matrix(arrZ, fK, fK);

        try {
            json jsonResult = {
                {"dtype", typeName},
                {"mnk", {fM, fN, fK}},
                {"seed", fSeed},
                {"range", {fMin, fMax}},
                {"tests", json::array()},
            };
            if (fCPU) {
                auto runtime = "C++";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto bench = GEMMFlops(matX, matY, matZ);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fOpenMP) {
                auto runtime = "OpenMP";
                std::cerr << "Running " << runtime << ".." << std::endl;

                MatrixOpenMP ompX(matX);
                auto bench = GEMMFlops(ompX, matY, matZ);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fOpenBLAS) {
                auto runtime = "OpenBLAS";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixBLAS(matX);
                auto y = MatrixBLAS(matY);
                auto z = MatrixBLAS(matZ);

                auto bench = GEMMFlops(x, y, z);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fOpenCL) {
                auto runtime = "OpenCL";

                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixOpenCL(matX);
                auto y = MatrixOpenCL(matY);
                auto z = MatrixOpenCL(matZ);

                auto bench = GEMMFlops(x, y, z);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fClBlast) {
                auto runtime = "CLBlast";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixCLBlast(matX);
                auto y = MatrixCLBlast(matY);
                auto z = MatrixCLBlast(matZ);

                auto bench = GEMMFlops(x, y, z);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            std::cout << jsonResult.dump(4) << std::endl;

            delete[] arrX;
            delete[] arrY;
            delete[] arrZ;

            std::cerr << "Finished" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }, sample);
}