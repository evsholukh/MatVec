#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "opencl.h"
#include "openblas.h"

#include "CLI11.hpp"


int main(int argc, char **argv) {

    CLI::App app{"MatMul"};
    std::string rows, cols, bs;

    app.add_option("-n,--rows", rows, "rows");
    app.add_option("-m,--cols", cols, "cols");
    app.add_option("-b,--block-size", bs, "block size");

    CLI11_PARSE(app, argc, argv);
    size_t N, M, blockSize;

    if (rows.empty()) {
        std::cerr << "N: ";
        std::cin >> N;
    } else {
        N = std::atoi(rows.c_str());
    }
    if (cols.empty()) {
        std::cerr << "M: ";
        std::cin >> M;
    } else {
        M = std::atoi(cols.c_str());
    }
    if (bs.empty()) {
        std::cerr << "Block Size: ";
        std::cin >> blockSize;
    } else {
        blockSize = std::atoi(bs.c_str());
    }

    try {
        std::cerr << "Creating array (size: " << N*M << ").." << std::endl;

        auto dataX = Utils::create_array<float>(N*M, blockSize, 0.0001f);
        auto dataY = Utils::create_array<float>(N*M, blockSize, 0.0001f);

        Utils::randomize_array(dataX, N*M);
        Utils::randomize_array(dataY, N*M);

        auto vx = Vector<float>(dataX, N*M);
        auto vy = Vector<float>(dataY, N*M);
        auto vbx = VectorBLAS(vx); 
        auto vby = VectorBLAS(vy);
        auto cl_vx = VectorCLBlast(vx);
        auto cl_vy = VectorCLBlast(vy);
        auto vrx = VectorOpenCL(vx, blockSize);

        std::cerr << "Memory size: " << vx.size_mb() + vy.size_mb() << "MB" << std::endl;

        printf("[");
        auto value = 0.0f;
        auto duration = Utils::measure([&vx, &vy, &value]() { value = vx.dot(vy); });
        printf("{\"duration\": %f, \"value\": %f, \"runtime\": \"%s\"},\n", duration, value, "C++");

        duration = Utils::measure([&vbx, &vby, &value]() { value = vbx.dot(vby); });
        printf("{\"duration\": %f, \"value\": %f, \"runtime\": \"%s\"},\n", duration, value, "OpenBLAS");

        duration = Utils::measure([&cl_vx, &cl_vy, &value]() { value = cl_vx.dot(cl_vy); });
        printf("{\"duration\": %f, \"value\": %f, \"runtime\": \"%s\"},\n", duration, value, "clBLASt");
    
        duration = Utils::measure([&vrx, &vy, &value]() { value = vrx.dot(vy); });
        printf("{\"duration\": %f, \"value\": %f, \"runtime\": \"%s\"}", duration, value, "OpenCL Reduction");
        printf("]");

        delete[] dataX;
        delete[] dataY;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}