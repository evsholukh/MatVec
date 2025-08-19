
#include <iostream>

#include "CLI11.hpp"
#include "utils.h"
#include "matrix.h"
#include "openblas.h"
#include "opencl.h"


int main(int argc, char **argv) {

    CLI::App app{"matrix"};
    std::string cols_str, rows_str;

    app.add_option("-c,--cols", cols_str, "cols");
    app.add_option("-r,--rows", rows_str, "cols");

    CLI11_PARSE(app, argc, argv);

    size_t cols = 512, rows = 1024;

    if (!cols_str.empty()) {
        cols = std::atoi(cols_str.c_str());
    }
    if (!rows_str.empty()) {
        rows = std::atoi(rows_str.c_str());
    }

    try {
        const size_t N = cols*rows, M = rows*rows;
        std::cerr << "Creating matrix.. (" << rows << "x" << cols << ")" << std::endl;

        auto arrX = Utils::create_array<float>(N, 1, 0.0001);
        Utils::randomize_array(arrX, N);

        auto arrY = Utils::create_array<float>(N, 1, 0.0001);
        Utils::randomize_array(arrY, N);

        auto arrZ = Utils::create_array<float>(M, 1, 0.0001);

        auto matX = Matrix<float>(arrX, rows, cols);
        auto matY = Matrix<float>(arrY, cols, rows);
        auto matZ = Matrix<float>(arrZ, rows, rows);

        printf("[");
        {
            auto duration = Utils::measure([&matX, &matY, &matZ]() { matX.dot(matY, matZ); });
            auto value = matZ.sum();

            printf("{\"duration\": %f,"
                    "\"sum\": %f,"
                    "\"rows\": %d,"
                    "\"cols\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"},\n", duration, value, rows, cols, "C++", "CPU");
        }
        {
            auto x = MatrixBLAS(matX);
            auto y = MatrixBLAS(matY);
            auto z = MatrixBLAS(matZ);

            auto duration = Utils::measure([&x, &y, &z]() { x.dot(y, z); });
            auto value = matZ.sum();

            printf("{\"duration\": %f,"
                    "\"sum\": %f,"
                    "\"rows\": %d,"
                    "\"cols\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"},\n", duration, value, rows, cols, "OpenBLAS", "CPU");
        }
        {
            auto x = MatrixCLBlast(matX);
            auto y = MatrixCLBlast(matY);
            auto z = MatrixCLBlast(matZ);

            auto duration = Utils::measure([&x, &y, &z]() { x.dot(y, z); });
            auto value = matZ.sum();

            printf("{\"duration\": %f,"
                    "\"sum\": %f,"
                    "\"rows\": %d,"
                    "\"cols\": %d,"
                    "\"runtime\": \"%s\","
                    "\"device\": \"%s\"}", duration, value, rows, cols, "clBLASt", "GPU");
        }
        printf("]");

        delete[] arrX;
        delete[] arrY;
        delete[] arrZ;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}