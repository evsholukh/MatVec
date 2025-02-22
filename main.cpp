#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

#include "vector.h"
#include "opencl.h"
#include "measured.h"
#include "experiments.h"


int main(int argc, char **argv) {
    const int N = pow(10, 7);

    try {
        auto x = Vector(N);
        auto cl_x = VectorOpenCL(x);

        cl_x.print_info();

        std::cout << "Vector size: " << x.size_mb() << "MB" << std::endl;
        std::cout << "Runtime sum: " << std::fixed << x.sum() << std::endl;
        std::cout << "OpenCL sum: " << std::fixed << cl_x.sum() << std::endl;

        auto y = Vector(N);
        auto cl_y = VectorOpenCL(y);

        std::cout << "Runtime dot: " << std::fixed << x.dot(y) << std::endl;
        std::cout << "OpenCL dot: " << std::fixed << cl_x.dot_cl(cl_y) << std::endl;

        x.add(y);
        cl_x.add(cl_y);
        std::cout << "MSE (add): " << std::fixed << x.mse(cl_x) << std::endl;

        // VectorAddOpenCL add(x.vec(), y.vec());
        // std::cout << std::left 
        //           << std::setw(20)
        //           << "OpenCL vector add: "
        //           << std::fixed
        //           << add.measure()
        //           << "s" << std::endl;

        // VectorSumOpenCL sum(x.vec());
        // std::cout << std::left 
        //           << std::setw(20)
        //           << "OpenCL vector sum: "
        //           << std::fixed
        //           << sum.measure()
        //           << "s" << std::endl;

        // VectorDotRuntime runtime_dot(x.vec(), y.vec());
        // std::cout << std::left 
        //           << std::setw(20)
        //           << "Runtime vector dot: "
        //           << std::fixed
        //           << sum.measure()
        //           << "s" << std::endl;

        // VectorDotOpenCL dot(x.vec(), y.vec());
        // std::cout << std::left 
        //           << std::setw(20)
        //           << "OpenCL vector dot: "
        //           << std::fixed
        //           << sum.measure()
        //           << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}