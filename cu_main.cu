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
#include "opencl.h"

typedef float vec_t;


int main(int argc, char **argv) {
    const size_t N = 4096;

    try {
        std::cout << "Randomization matrix.. (size: " << std::to_string(N) << "x" << std::to_string(N) << ")" << std::endl;

        auto mat_x = Matrix<vec_t>::random(N, N);
        auto mat_y = Matrix<vec_t>::random(N, N);

        std::cout << "Matrix size: " << mat_x.size_mb() << "MB" << std::endl;

        MatrixOpenCL cl_mat_x(mat_x), cl_mat_y(mat_y);
        MatrixCuda<vec_t> cuda_mat_x(mat_x), cuda_mat_y(mat_y);

        MatrixSum mat_sum(mat_x);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime matrix sum: "
                  << std::fixed
                  << mat_sum.measure()
                  << "s" << std::endl;

        MatrixSum mat_cl_sum(cl_mat_x);
        std::cout << std::left 
                << std::setw(20)
                << "OpenCL matrix sum: "
                << std::fixed
                << mat_cl_sum.measure()
                << "s" << std::endl;

        MatrixSum mat_cuda_sum(cuda_mat_x);
        std::cout << std::left 
                << std::setw(20)
                << "CUDA matrix sum: "
                << std::fixed
                << mat_cuda_sum.measure()
                << "s" << std::endl;

        MatrixAdd mat_add(mat_x, mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "Runtime matrix add: "
                << std::fixed
                << mat_add.measure()
                << "s" << std::endl;

        MatrixAdd mat_cl_add(cl_mat_x, cl_mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "OpenCL matrix add: "
                << std::fixed
                << mat_cl_add.measure()
                << "s" << std::endl;

        MatrixAdd<vec_t> mat_cuda_add(cuda_mat_x, cuda_mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "CUDA matrix add: "
                << std::fixed
                << mat_cuda_add.measure()
                << "s" << std::endl;

        // MatrixMul mat_mul(mat_x, mat_y);
        // std::cout << std::left 
        //         << std::setw(20)
        //         << "Runtime matrix mul: "
        //         << std::fixed
        //         << mat_mul.measure()
        //         << "s" << std::endl;

        MatrixMul mat_cl_mul(cl_mat_x, cl_mat_y);
        std::cout << std::left 
                << std::setw(20)
                << "OpenCL matrix mul: "
                << std::fixed
                << mat_cl_mul.measure()
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

    return EXIT_SUCCESS;
}