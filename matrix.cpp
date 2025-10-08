
#include <iostream>
#include <variant>

#include "utils.h"
#include "matrix.h"
#include "openblas.h"
#include "opencl.h"
#include "openmp.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    int fCols = 1000, fRows = 1000;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;

    bool fCPU = false,
         fOpenMP = false,
         fOpenBLAS = false,
         fClBlast = false,
         fFloat = false,
         fDouble = false, 
         fAll = false;

    app.add_option("-c,--cols", fCols, "cols");
    app.add_option("-r,--rows", fRows, "rows");
    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cpu", fCPU, "CPU");
    app.add_flag("--openmp", fOpenMP, "OpenMP");
    app.add_flag("--openblas", fOpenBLAS, "OpenBLAS");
    app.add_flag("--clblast", fClBlast, "CLBlast");

    app.add_flag("-a,--all", fAll, "All");
    app.add_flag("--float", fFloat, "use float type");
    app.add_flag("--double", fDouble, "use double type");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fCPU = true;
        fOpenMP = true;
        fOpenBLAS = true;
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

        const size_t N = fCols*fRows, M = fRows*fRows;

        std::cerr << "Creating array " << N << ".." << std::endl;
        auto arrX = Utils::create_array<T>(N, 1);
        Utils::randomize_array<T>(arrX, N, fMin, fMax, fSeed);
        auto matX = Matrix(arrX, fRows, fCols);
        std::cerr << "Memory utilized: " << matX.size_mb() << "MB" << std::endl;

        std::cerr << "Creating array " << N << ".." << std::endl;
        auto arrY = Utils::create_array<T>(N, 1);
        Utils::randomize_array<T>(arrY, N, fMin, fMax, fSeed);
        auto matY = Matrix(arrY, fCols, fRows);
        std::cerr << "Memory utilized: " << matY.size_mb() << "MB" << std::endl;

        std::cerr << "Creating array " << M << ".." << std::endl;
        auto arrZ = Utils::create_array<T>(M, 1);
        auto matZ = Matrix(arrZ, fRows, fRows);
        std::cerr << "Memory utilized: " << matZ.size_mb() << "MB" << std::endl;

        try {
            json jsonResult = {
                {"type", typeName},
                {"rows", fRows},
                {"cols", fCols},
                {"seed", fSeed},
                {"range", {fMin, fMax}},
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
            if (fOpenMP) {
                auto runtime = "OpenMP";
                std::cerr << "Running " << runtime << ".." << std::endl;

                MatrixOpenMP ompX(matX);

                auto duration = Utils::measure([&ompX, &matY, &matZ]() { ompX.dot(matY, matZ); });
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
    }, sample);
}