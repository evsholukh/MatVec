#include <iostream>
#include <random>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

#include "vector.h"
#include "cl.h"
#include "measured.h"

using namespace std;


int main(int argc, char **argv) {
    const int array_size = pow(10, 7);

    Vector a = Vector::generate(array_size);
    Vector b = Vector::generate(array_size);

    std::cout << "Vector size: " << a.size_mb() << " MB" << std::endl;
    try {
        OpenCLHelper helper;
        helper.print_info();
        helper.vector_sum(a.data(), b.data());
        helper.close();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    std::cout << "Finished" << std::endl;

    return 0;
}