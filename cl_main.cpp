#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

#include "matrix.h"
#include "vector.h"
#include "opencl.h"
#include "measured.h"
#include "experiments.h"

typedef float vec_t;


int main(int argc, char **argv) {

    const size_t N = 1024;

    try {
        std::cout << "Randomization matrix.. (size: " << std::to_string(N) << "x" << std::to_string(N) << ")" << std::endl;

        auto mat_x = Matrix<vec_t>::random(N, N);
        auto mat_y = Matrix<vec_t>::random(N, N);

        MatrixOpenCL cl_mat_x(mat_x), cl_mat_y(mat_y);

        std::cout << "Matrix size: " << mat_x.size_mb() << "MB" << std::endl;

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

        std::cout << "Randomization vector.. (size: " << std::to_string(N) << ")" << std::endl;

        auto vec_x = Vector<vec_t>::random(N);
        auto vec_y = Vector<vec_t>::random(N);

        VectorOpenCL<vec_t> cl_vec_x(vec_x), cl_vec_y(vec_y);

        std::cout << "Vector size: " << vec_x.size_mb() << "MB" << std::endl;

        VectorAdd vec_add(vec_x, vec_y);
        std::cout << std::left
                << std::setw(20)
                << "Runtime vector add: "
                << std::fixed
                << vec_add.measure()
                << "s" << std::endl;

        VectorAdd vec_cl_add(cl_vec_x, cl_vec_y);
        std::cout << std::left
                << std::setw(20)
                << "OpenCL vector add: "
                << std::fixed
                << vec_cl_add.measure()
                << "s" << std::endl;

        VectorSum vec_sum(vec_x);
        std::cout << std::left
                << std::setw(20)
                << "Runtime vector sum: "
                << std::fixed
                << vec_sum.measure()
                << "s" << std::endl;

        VectorSum vec_cl_sum(cl_vec_x);
        std::cout << std::left
                << std::setw(20)
                << "OpenCL vector sum: "
                << std::fixed
                << vec_cl_sum.measure()
                << "s" << std::endl;

        VectorMul vec_dot(vec_x, vec_y);
        std::cout << std::left
                << std::setw(20)
                << "Runtime vector dot: "
                << std::fixed
                << vec_dot.measure()
                << "s" << std::endl;

        VectorMul vec_cl_dot(cl_vec_x, cl_vec_y);
        std::cout << std::left
                << std::setw(20)
                << "OpenCL vector dot: "
                << std::fixed
                << vec_cl_dot.measure()
                << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}