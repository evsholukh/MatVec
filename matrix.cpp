
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

    int fCols = 1000,
        fRows = 1000;

    bool fCPU = false,
         fOpenBLAS = false,
         fClBlast = false,
         fAll = false;

    app.add_option("-c,--cols", fCols, "cols");
    app.add_option("-r,--rows", fRows, "cols");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--clblast", fClBlast, "clBLASt");
    app.add_flag("-a,--all", fAll, "All");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCPU = true;
        fOpenBLAS = true;
        fClBlast = true;
    }

    try {
        const size_t N = fCols*fRows, M = fRows*fRows;
        
        std::cerr << "Creating matrix " << fRows << "x" << fCols << ".." << std::endl;
        auto arrX = Utils::create_array<float>(N, 1);

        std::cerr << "Creating matrix " << fCols << "x" << fRows << ".." << std::endl;
        auto arrY = Utils::create_array<float>(N, 1);

        std::cerr << "Creating matrix " << fRows << "x" << fRows << ".." << std::endl;
        auto arrZ = Utils::create_array<float>(M, 1);

        Utils::randomize_array(arrX, N);
        Utils::randomize_array(arrY, N);

        auto matX = Matrix<float>(arrX, fRows, fCols);
        auto matY = Matrix<float>(arrY, fCols, fRows);
        auto matZ = Matrix<float>(arrZ, fRows, fRows);

        std::cerr << "Memory utilized: " << matX.size_mb() + matY.size_mb() + matZ.size_mb() << "MB" << std::endl;

        std::cerr << "Running control.." << std::endl;
        matX.dot(matY, matZ);
        auto control = matZ.sum();

        auto fO3 = false;
        #ifdef OPT_LEVEL_O3
            fO3 = true;
        #endif
        json jsonResult = {
            {"rows", fRows},
            {"cols", fCols},
            {"control", control},
            {"cpu", Utils::cpuName().c_str()},
            {"gpu", OpenCL::deviceName(OpenCL::defaultDevice()).c_str()},
            {"o3", fO3},
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
            auto runtime = "clBLASt";
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
        std::cout << jsonResult.dump(4);

        delete[] arrX;
        delete[] arrY;
        delete[] arrZ;

        std::cerr << std::endl;
        std::cerr << "Finished" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}