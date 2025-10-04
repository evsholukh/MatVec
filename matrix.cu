
#include <iostream>

#include "utils.h"
#include "matrix.h"
#include "cuda.cuh"
#include "opencl.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{"matrix"};

    int fCols = 1000,
        fRows = 1000,
        fSeed = std::chrono::system_clock::now().time_since_epoch().count();

    float fMin = -1.0,
          fMax = 1.0;

    bool fcuBLAS = false,
         fAll = false;

    app.add_option("-c,--cols", fCols, "cols");
    app.add_option("-r,--rows", fRows, "rows");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cublas", fcuBLAS, "cuBLAS");

    app.add_flag("-a,--all", fAll, "All");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fcuBLAS = true;
    }
    const size_t N = fCols*fRows, M = fRows*fRows;
    
    std::cerr << "Creating array " << N << ".." << std::endl;
    auto arrX = Utils::create_array<float>(N, 1);
    Utils::randomize_array(arrX, N, fMin, fMax, fSeed);
    auto matX = Matrix<float>(arrX, fRows, fCols);
    std::cerr << "Memory utilized: " << matX.size_mb() << "MB" << std::endl;

    std::cerr << "Creating array " << N << ".." << std::endl;
    auto arrY = Utils::create_array<float>(N, 1);
    Utils::randomize_array(arrY, N, fMin, fMax, fSeed);
    auto matY = Matrix<float>(arrY, fCols, fRows);
    std::cerr << "Memory utilized: " << matY.size_mb() << "MB" << std::endl;

    std::cerr << "Creating array " << M << ".." << std::endl;
    auto arrZ = Utils::create_array<float>(M, 1);
    auto matZ = Matrix<float>(arrZ, fRows, fRows);
    std::cerr << "Memory utilized: " << matZ.size_mb() << "MB" << std::endl;

    try {
        std::cerr << "Running.." << std::endl;
        matX.dot(matY, matZ);
        auto result = matZ.sum();

        auto fO3 = false;
        #ifdef OPT_LEVEL_O3
            fO3 = true;
        #endif
        json jsonResult = {
            {"rows", fRows},
            {"cols", fCols},
            {"result", result},
            {"seed", fSeed},
            {"min", fMin},
            {"max", fMax},
            {"cpu", Utils::cpuName().c_str()},
            {"gpu", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()},
            {"o3", fO3},
            {"tests", json::array()},
        };
        if (fcuBLAS) {
            auto runtime = "cuBLAS";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto x = MatrixCuda(matX);
            auto y = MatrixCuda(matY);
            auto z = MatrixCuda(matZ);

            auto duration = Utils::measure([&x, &y, &z]() { x.dot(y, z); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", matZ.sum()},
                {"runtime", runtime},
            });
        }
        std::cout << jsonResult.dump(4);

        delete[] arrX;
        delete[] arrY;
        delete[] arrZ;

        std::cerr << "Finished" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}