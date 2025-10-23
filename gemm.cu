
#include <iostream>
#include <variant>

#include "utils.h"
#include "matrix.h"

#include "vector.cuh"
#include "bench.h"

#include "CLI11.hpp"
#include "json.hpp"

using json = nlohmann::json;


int main(int argc, char **argv) {

    CLI::App app{argv[0]};

    auto fM = 1000, fN = 1000, fK = 1000;

    auto fSeed = std::chrono::system_clock::now().time_since_epoch().count();
    auto fMin = -1.0, fMax = 1.0;

    bool fcuBLAS = false,
        fCUDA = false,
        fFloat = false,
        fDouble = false,
        fAll = false;

    app.add_option("-m", fM, "x-rows");
    app.add_option("-n", fN, "y-cols");
    app.add_option("-k", fK, "x-cols, y-rows");

    app.add_option("-s,--seed", fSeed, "random seed");

    app.add_option("--low", fMin, "random lower value");
    app.add_option("--high", fMax, "random higher value");

    app.add_flag("--cublas", fcuBLAS, "cuBLAS");
    app.add_flag("--cuda", fCUDA, "CUDA");
    app.add_flag("-a,--all", fAll, "All");

    app.add_flag("--float", fFloat, "single precision");
    app.add_flag("--double", fDouble, "double precision");

    CLI11_PARSE(app, argc, argv);

    if (fAll) {
        fcuBLAS = true;
        fCUDA = true;
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

        auto arrX = Utils::create_array<T>(fM*fK, 1);
        auto arrY = Utils::create_array<T>(fK*fN, 1);
        auto arrZ = Utils::create_array<T>(fK*fK, 1);

        Utils::randomize_array<T>(arrX, fM*fK, fMin, fMax, fSeed);
        Utils::randomize_array<T>(arrY, fK*fN, fMin, fMax, fSeed);

        auto matX = Matrix<T>(arrX, fM, fK);
        auto matY = Matrix<T>(arrY, fK, fN);
        auto matZ = Matrix<T>(arrZ, fK, fK);

        try {
            json jsonResult = {
                {"type", typeName},
                {"M", fM},
                {"N", fN},
                {"K", fK},
                {"seed", fSeed},
                {"range", {fMin, fMax}},
                {"cpu", Utils::cpuName()},
                {"gpu", CUDA::getDeviceName()},
                {"tests", json::array()},
            };
            if (fcuBLAS) {
                auto runtime = "cuBLAS";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixCuBLAS(matX);
                auto y = MatrixCuBLAS(matY);
                auto z = MatrixCuBLAS(matZ);

                auto bench = GEMMFlops(x, y, z);
                auto metric = bench.perform();

                jsonResult["tests"].push_back({
                    {"gflops", metric.gflops()},
                    {"result", metric.result()},
                    {"runtime", runtime},
                });
            }
            if (fCUDA) {
                auto runtime = "CUDA";
                std::cerr << "Running " << runtime << ".." << std::endl;

                auto x = MatrixCUDA(matX);
                auto y = MatrixCUDA(matY);
                auto z = MatrixCUDA(matZ);

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