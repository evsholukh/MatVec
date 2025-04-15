#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

#include "matrix.h"
#include "vector.h"
#include "measured.h"
#include "experiments.h"
#include "cuda.cuh"

typedef float vec_t;


int main(int argc, char **argv) {

    const size_t N = 1024;

    try {
        std::cout << "Randomization matrix.. (size: " << std::to_string(N) << "x" << std::to_string(N) << ")" << std::endl;

        Matrix<vec_t> mat_x = Matrix<vec_t>::random(N, N);
        Matrix<vec_t> mat_y = Matrix<vec_t>::random(N, N);

        MatrixCuda<vec_t> cuda_mat_x(mat_x), cuda_mat_y(mat_y);

        std::cout << "Matrix size: " << mat_x.size_mb() << "MB" << std::endl;

        MatrixSum<vec_t> mat_sum(mat_x);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix sum: "
                  << std::fixed
                  << mat_sum.measure()
                  << "s" << std::endl;

        MatrixSum<vec_t> mat_cuda_sum(cuda_mat_x);
        std::cout << std::left 
                << std::setw(20)
                << "CUDA matrix sum: "
                << std::fixed
                << mat_cuda_sum.measure()
                << "s" << std::endl;

        MatrixAdd<vec_t> mat_add(mat_x, mat_y);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix add: "
                  << std::fixed
                  << mat_add.measure()
                  << "s" << std::endl;

        MatrixAdd<vec_t> mat_cuda_add(cuda_mat_x, cuda_mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "CUDA matrix add: "
                << std::fixed
                << mat_cuda_add.measure()
                << "s" << std::endl;

        MatrixMul<vec_t> mat_mul(mat_x, mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "Runtime matrix mul: "
                << std::fixed
                << mat_mul.measure()
                << "s" << std::endl;

        MatrixMul<vec_t> mat_cuda_mul(cuda_mat_x, cuda_mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "CUDA matrix mul: "
                << std::fixed
                << mat_cuda_mul.measure()
                << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}