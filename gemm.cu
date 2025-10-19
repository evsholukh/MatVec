
#include <iostream>
#include <variant>

#include "utils.h"
#include "matrix.h"

#include "vector.cuh"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    int fCols = 1000, fRows = 1000;
    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    float fMin = -1.0, fMax = 1.0;
    bool fcuBLAS = false,
         fFloat = false,
         fDouble = false,
         fAll = false;

    app.add_option("-c,--cols", fCols, "cols");
    app.add_option("-r,--rows", fRows, "rows");
    app.add_option("-s,--seed", fSeed, "random seed");
    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cublas", fcuBLAS, "cuBLAS");
    app.add_flag("-a,--all", fAll, "All");
    app.add_flag("--float", fFloat, "use float type");
    app.add_flag("--double", fDouble, "use double type");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
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

        const size_t N = fCols*fRows,
                     M = fRows*fRows;

        std::cerr << "Creating array " << N << ".." << std::endl;
        auto arrX = Utils::create_array<T>(N, 1);
        Utils::randomize_array<T>(arrX, N, fMin, fMax, fSeed);
        auto matX = Matrix<T>(arrX, fRows, fCols);
        std::cerr << "Memory utilized: " << matX.size_mb() << "MB" << std::endl;
    
        std::cerr << "Creating array " << N << ".." << std::endl;
        auto arrY = Utils::create_array<T>(N, 1);
        Utils::randomize_array<T>(arrY, N, fMin, fMax, fSeed);
        auto matY = Matrix<T>(arrY, fCols, fRows);
        std::cerr << "Memory utilized: " << matY.size_mb() << "MB" << std::endl;
    
        std::cerr << "Creating array " << M << ".." << std::endl;
        auto arrZ = Utils::create_array<T>(M, 1);
        auto matZ = Matrix<T>(arrZ, fRows, fRows);
        std::cerr << "Memory utilized: " << matZ.size_mb() << "MB" << std::endl;

        try {
            json jsonResult = {
                {"type", typeName},
                {"rows", fRows},
                {"cols", fCols},
                {"seed", fSeed},
                {"range", {fMin, fMax}},
                {"cpu", Utils::cpuName()},
                {"gpu", CUDA::getDeviceName()},
                {"optimization", Utils::getOptimizationFlag()},
                {"tests", json::array()},
            };
            if (fcuBLAS) {
                auto runtime = "cuBLAS";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixCuBLAS(matX);
                auto y = MatrixCuBLAS(matY);
                auto z = MatrixCuBLAS(matZ);

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

    }, sample);
}