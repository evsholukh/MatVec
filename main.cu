#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "opencl.h"
#include "utils.h"
#include "cuda.cuh"


template<typename T>
T* random_vector(const size_t size) {
    T *data = new T[size];

    std::mt19937 generator(42);
    std::uniform_real_distribution<T> dist(-1, 1);

    for (size_t i = 0; i < size; i++) {
        data[i] = dist(generator);
    }
    return data;
}

int main(int argc, char **argv) {

    const size_t N = 1024, M = 1024;

    try {
        std::cout << "Randomization data.." << std::endl;

        float *data_x = random_vector<float>(N*M);
        float *data_y = random_vector<float>(N*M);
        float *data_z = random_vector<float>(N*M);

        Vector<float> vx(data_x, N*M);
        Vector<float> vy(data_x, N*M);

        Matrix<float> mx(data_x, N, M);
        Matrix<float> my(data_y, N, M);
        Matrix<float> mz(data_z, N, M);

        MatrixOpenCL cl_mx(mx), cl_my(my), cl_mz(mz);

        VectorCuda cuda_vx(vx), cuda_vy(vy);
        MatrixCuda cuda_mx(mx), cuda_my(my), cuda_mz(mz);

        std::cout << "Matrix size: " << mx.size_mb() << "MB" << std::endl;

        // std::cout << std::left 
        //           << std::setw(20)
        //           << "C++ vector dot: "
        //           << std::fixed
        //           << Utils::measure([&vx, &vy]() {
        //               std::cout << "(" << vx.dot(vy) << ")" << " ";
        //           })
        //           << "s" << std::endl;

        // std::cout << std::left
        //           << std::setw(20)
        //           << "cuBLAS vector dot: "
        //           << std::fixed
        //           << Utils::measure([&cuda_vx, &cuda_vy]() {
        //               std::cout << "(" << cuda_vx.dot(cuda_vy) << ")" << " ";
        //           })
        //           << "s" << std::endl;

        // std::cout << std::left 
        //           << std::setw(20)
        //           << "C++ matrix mul: "
        //           << std::fixed
        //           << Utils::measure([&mx, &my, &mz]() {
        //                 mx.dot(my, mz);
        //                 std::cout << "(" << mz.sum() << ")" << " ";
        //           })
        //           << "s" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "clBLAST matrix mul: "
                  << std::fixed
                  << Utils::measure([&cl_mx, &cl_my, &cl_mz]() {
                        cl_mx.dot(cl_my, cl_mz);
                        std::cout << "(" << cl_mz.sum() << ")" << " ";
                  })
                  << "s" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "cuBLAS matrix mul: "
                  << std::fixed
                  << Utils::measure([&cuda_mx, &cuda_my, &cuda_mz]() {
                        cuda_mx.dot(cuda_my, cuda_mz);
                        std::cout << "(" << cuda_mz.sum() << ")" << " ";
                  })
                  << "s" << std::endl;

        delete[] data_x;
        delete[] data_y;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}