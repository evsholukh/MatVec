#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "opencl.h"
#include "measured.h"
#include "experiments.h"


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

    const size_t N = 1024;
    const size_t M = 1024;

    try {
        std::cout << "Randomization data.." << std::endl;

        float *data_x = random_vector<float>(N*M);
        float *data_y = random_vector<float>(N*M);

        Matrix<float> mx(data_x, N, M);
        Matrix<float> my(data_y, N, M);

        MatrixOpenCL cl_mx(mx), cl_my(my);

        std::cout << "Matrix size: " << mx.size_mb() << "MB" << std::endl;

        MatrixSum mat_sum(mx);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix sum: "
                  << std::fixed
                  << mat_sum.measure()
                  << "s" << std::endl;

        MatrixSum mat_cl_sum(cl_mx);
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL matrix sum: "
                  << std::fixed
                  << mat_cl_sum.measure()
                  << "s" << std::endl;

        MatrixAdd mat_add(mx, my);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix add: "
                  << std::fixed
                  << mat_add.measure()
                  << "s" << std::endl;

        MatrixAdd mat_cl_add(cl_mx, cl_my);
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL matrix add: "
                  << std::fixed
                  << mat_cl_add.measure()
                  << "s" << std::endl;

        MatrixMul mat_mul(mx, my);
        std::cout << std::left 
                << std::setw(20)
                << "Runtime matrix mul: "
                << std::fixed
                << mat_mul.measure()
                << "s" << std::endl;

        MatrixMul mat_cl_mul(cl_mx, cl_my);
        std::cout << std::left 
                << std::setw(20)
                << "OpenCL matrix mul: "
                << std::fixed
                << mat_cl_mul.measure()
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