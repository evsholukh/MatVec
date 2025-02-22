#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

#include "vector.h"
#include "cl.h"
#include "measured.h"
#include "experiments.h"


int main(int argc, char **argv) {
    const int N = pow(10, 8);

    auto x = Vector::generate(N);
    auto y = Vector::generate(N);

    std::cout << "Vector size mb: " << x.size_mb() << std::endl;
    try {
        VectorAddOpenCL add(x.data(), y.data());
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL vector add: "
                  << std::fixed
                  << add.measure()
                  << "s" << std::endl;

        VectorSumOpenCL sum(x.data());
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL vector sum: "
                  << std::fixed
                  << sum.measure()
                  << "s" << std::endl;

        VectorDotRuntime runtime_dot(x.data(), y.data());
        std::cout << std::left 
                  << std::setw(20)
                  << "Runtime vector dot: "
                  << std::fixed
                  << sum.measure()
                  << "s" << std::endl;

        VectorDotOpenCL dot(x.data(), y.data());
        std::cout << std::left 
                  << std::setw(20)
                  << "OpenCL vector dot: "
                  << std::fixed
                  << sum.measure()
                  << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}