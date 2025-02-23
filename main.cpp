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
    const int N = pow(10, 8);

    try {
        std::cout << "Generating random vector.. (size: " << std::to_string(N) << ")" << std::endl;
        Vector x = Vector<double>::random(N);
        std::cout << "Vector size: " << x.size_mb() << "MB" << std::endl;

        Vector y = Vector<double>::random(N);

        Vector cl_x = VectorOpenCL(x);
        Vector cl_y = VectorOpenCL(y);

        VectorAdd add(x, y);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime vector add: "
                  << std::fixed
                  << add.measure()
                  << "s" << std::endl;

        VectorAdd cl_add(cl_x, cl_y);
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL vector add: "
                  << std::fixed
                  << cl_add.measure()
                  << "s" << std::endl;

        VectorSum sum(x);
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime vector sum: "
                  << std::fixed
                  << sum.measure()
                  << "s" << std::endl;

        VectorSum cl_sum(cl_x);
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL vector sum: "
                  << std::fixed
                  << cl_sum.measure()
                  << "s" << std::endl;

        VectorDot dot(x, y);
        std::cout << std::left 
                << std::setw(20)
                << "Runtime vector dot: "
                << std::fixed
                << dot.measure()
                << "s" << std::endl;

        VectorDot cl_dot(cl_x, cl_y);
        std::cout << std::left 
                << std::setw(20)
                << "OpenCL vector dot: "
                << std::fixed
                << cl_dot.measure()
                << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}