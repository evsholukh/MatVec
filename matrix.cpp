
#include <iostream>

#include "utils.h"
#include "matrix.h"
#include "openblas.h"
#include "opencl.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{"matrix"};

    int fCols = 1000, fRows = 1000;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCPU = false,
         fOpenBLAS = false,
         fClBlast = false,
         fAll = false;

    app.add_option("-c,--cols", fCols, "cols");
    app.add_option("-r,--rows", fRows, "rows");

    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--clblast", fClBlast, "CLBlast");

    app.add_flag("-a,--all", fAll, "All");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCPU = true;
        fOpenBLAS = true;
        fClBlast = true;
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

        json jsonResult = {
            {"rows", fRows},
            {"cols", fCols},
            {"result", result},
            {"seed", fSeed},
            {"min", fMin},
            {"max", fMax},
            {"tests", json::array()},
        };
        if (fCPU) {
            auto runtime = "C++";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto duration = Utils::measure([&matX, &matY, &matZ]() { matX.dot(matY, matZ); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", matZ.sum()},
                {"runtime", runtime},
            });
        }
        if (fOpenBLAS) {
            auto runtime = "OpenBLAS";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto x = MatrixBLAS(matX);
            auto y = MatrixBLAS(matY);
            auto z = MatrixBLAS(matZ);

            auto duration = Utils::measure([&x, &y, &z]() { x.dot(y, z); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", matZ.sum()},
                {"runtime", runtime},
            });
        }
        if (fClBlast) {
            auto runtime = "CLBlast";
            std::cerr << "Running " << runtime << ".." << std::endl;

            auto x = MatrixCLBlast(matX);
            auto y = MatrixCLBlast(matY);
            auto z = MatrixCLBlast(matZ);

            auto duration = Utils::measure([&x, &y, &z]() { x.dot(y, z); });
            jsonResult["tests"].push_back({
                {"duration", duration},
                {"result", matZ.sum()},
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
}