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

    const size_t N = 10;
    const size_t M = 10;

    try {
        std::cout << "Randomization data.." << std::endl;

        float *data_x = random_vector<float>(N*M);
        float *data_y = random_vector<float>(N*M);

        Matrix<float> mx(data_x, N, M);
        Matrix<float> my(data_y, N, M);

        MatrixOpenCL cl_mx(mx), cl_my(my);
        MatrixCuda cuda_mx(mx), cuda_my(my);

        std::cout << "Matrix size: " << mx.size_mb() << "MB" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix dot: "
                  << std::fixed
                  << Utils::measure([&mx, &my]() {
                      std::cout << "(" << (mx.dot(my)).sum() << ")" << " ";
                  })
                  << "s" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL matrix dot: "
                  << std::fixed
                  << Utils::measure([&cl_mx, &cl_my]() {
                      std::cout << "(" << (cl_mx.dot(cl_my)).sum() << ")" << " ";
                  })
                  << "s" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "CUDA matrix dot: "
                  << std::fixed
                  << Utils::measure([&cuda_mx, &cuda_my]() {
                      std::cout << "(" << (cuda_mx.dot(cuda_my)).sum() << ")" << " ";
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