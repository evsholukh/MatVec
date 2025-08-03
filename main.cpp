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
    
    size_t N, M, blockSize;

    CLI::App app{"MatMul"};
    std::string rows, cols, bs;
    bool useMatrix;

    app.add_option("-n,--rows", rows, "rows");
    app.add_option("-m,--cols", cols, "cols");
    app.add_option("-b,--block-size", bs, "block size");

    app.add_flag("--matrix", useMatrix, "use matrix");

    CLI11_PARSE(app, argc, argv);

    if (rows.empty()) {
        std::cout << "N: ";
        std::cin >> N;
    } else {
        N = std::atoi(rows.c_str());
    }
    if (cols.empty()) {
        std::cout << "M: ";
        std::cin >> M;
    } else {
        M = std::atoi(cols.c_str());
    }
    if (bs.empty()) {
        std::cout << "Block Size: ";
        std::cin >> blockSize;
    } else {
        blockSize = std::atoi(bs.c_str());
    }

    try {
        std::cout << "Creating array (size: " << N*M << ").." << std::endl;

        float *dataX = Utils::create_array<float>(N*M, blockSize, 0.0001f);
        float *dataY = Utils::create_array<float>(N*M, blockSize, 0.0001f);
        float *dataZ = Utils::create_array<float>(N*N, blockSize, 0.0f);

        Utils::randomize_array(dataX, N*M);
        Utils::randomize_array(dataY, N*M);

        Vector<float> vx(dataX, N*M), vy(dataY, N*M);
        VectorBLAS vbx(vx), vby(vy);
        VectorCLBlast cl_vx(vx), cl_vy(vy);
        VectorOpenCL vrx(vx, blockSize);

        Matrix<float> mx(dataX, N, M), my(dataY, M, N), mz(dataZ, N, N);
        MatrixBLAS mbx(mx), mby(my), mbz(mz);
        MatrixCLBlast cl_mx(mx), cl_my(my), cl_mz(mz);

        std::cout << "Memory size: " << mx.size_mb() + my.size_mb() + mz.size_mb() << "MB" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "C++ vector dot: "
                << std::fixed
                << Utils::measure([&vx, &vy]() {
                    std::cout << "(" << vx.dot(vy) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "OpenBLAS vector dot: "
                << std::fixed
                << Utils::measure([&vbx, &vby]() {
                    std::cout << "(" << vbx.dot(vby) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "clBLASt vector dot: "
                << std::fixed
                << Utils::measure([&cl_vx, &cl_vy]() {
                    std::cout << "(" << cl_vx.dot(cl_vy) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "OpenCL reduction vector dot: "
                << std::fixed
                << Utils::measure([&vrx, &vy]() {
                    std::cout << "(" << vrx.dot(vy) << ")" << " ";
                })
                << "s" << std::endl;

        if (useMatrix) {
            std::cout << std::left
                    << std::setw(20)
                    << "C++ matrix mul: "
                    << std::fixed
                    << Utils::measure([&mx, &my, &mz]() {
                        mx.dot(my, mz);
                        std::cout << "(" << mz.sum() << ")" << " ";
                    })
                    << "s" << std::endl;
    
            std::cout << std::left 
                    << std::setw(20)
                    << "OpenBLAS matrix mul: "
                    << std::fixed
                    << Utils::measure([&mbx, &mby, &mbz]() {
                        mbx.dot(mby, mbz);
                        std::cout << "(" << mbz.sum() << ")" << " ";
                    })
                    << "s" << std::endl;
    
            std::cout << std::left
                    << std::setw(20)
                    << "clBLASt matrix mul: "
                    << std::fixed
                    << Utils::measure([&cl_mx, &cl_my, &cl_mz]() {
                        cl_mx.dot(cl_my, cl_mz);
                        std::cout << "(" << cl_mz.sum() << ")" << " ";
                    })
                    << "s" << std::endl;
                }

        delete[] dataX;
        delete[] dataY;
        delete[] dataZ;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}