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
    const int N = pow(10, 7);

    try {
        VectorAdd add(N);
        float add_time = add.measure();
        std::cout << std::left << std::setw(20) << "OpenCL vector add: " << std::fixed << add_time << "s" << std::endl;

        VectorSum sum(N);
        float sum_time = sum.measure();
        std::cout << std::left << std::setw(20) << "OpenCL vector sum: " << std::fixed << sum_time << "s" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}