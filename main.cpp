#include <iostream>
#include <random>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

#include "vector.h"
#include "cl.h"
#include "measured.h"


int main(int argc, char **argv) {
    const int array_size = pow(10, 7);

    Vector a = Vector::generate(array_size);
    // Vector a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    Vector b = Vector::generate(array_size);

    std::cout << "Vector size: " << a.size_mb() << " MB" << std::endl;

    try {
        OpenCLHelper helper;
        helper.print_info();
        // helper.vector_add(a.data(), b.data());
        std::cout << "Runtime vector sum: " << std::fixed << a.sum() << std::endl;

        auto cl_vector_sum = helper.vector_sum(a.data());
        std::cout << "OpenCL vector sum: " << std::fixed << cl_vector_sum << std::endl;

        helper.close();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}