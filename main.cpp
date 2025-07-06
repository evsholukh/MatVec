#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"

#include "opencl.h"
#include "openblas.h"


int main(int argc, char **argv) {

    size_t N, M;

    std::cout << "N: ";
    std::cin >> N;

    std::cout << "M: ";
    std::cin >> M;

    try {
        std::cout << "Creating array (size: " << N*M << ").." << std::endl;

        auto platform = OpenCL::defaultPlatform();
        auto device = OpenCL::defaultDevice(platform);
        auto group_size = OpenCL::maxGroupSize(device);

        float *data_x = Utils::create_array<float>(N*M, group_size, 0.000001f);
        float *data_y = Utils::create_array<float>(N*M, group_size, 0.000001f);
        float *data_z = Utils::create_array<float>(N*N, group_size, 0.000001f);

        Vector<float> vx(data_x, N*M), vy(data_x, N*M);
        VectorBLAS vbx(vx), vby(vy);
        VectorCLBlast cl_vx(vx, device), cl_vy(vy, device);
        VectorReductionOpenCL vrx(vx, device);

        Matrix<float> mx(data_x, N, M), my(data_y, M, N), mz(data_z, N, N);
        MatrixBLAS mbx(mx), mby(my), mbz(mz);
        MatrixCLBlast cl_mx(mx, device), cl_my(my, device), cl_mz(mz, device);

        std::cout << "Memory size: " << mx.size_mb() << "MB" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "OpenBLAS vector dot: "
                << std::fixed
                << Utils::measure([&vbx, &vby]() {
                    std::cout << "(" << vbx.dot(vby) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "clBLASt vector dot: "
                << std::fixed
                << Utils::measure([&cl_vx, &cl_vy]() {
                    std::cout << "(" << cl_vx.dot(cl_vy) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "OpenCL reduction vector dot: "
                << std::fixed
                << Utils::measure([&vrx, &vy]() {
                    std::cout << "(" << vrx.dot(vy) << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left 
                << std::setw(20)
                << "OpenBLAS matrix mul: "
                << std::fixed
                << Utils::measure([&mbx, &mby, &mbz]() {
                    mbx.dot(mby, mbz);
                    std::cout << "(" << mbz.sum() << ")" << " ";
                })
                << "s" << std::endl;

        std::cout << std::left
                << std::setw(20)
                << "clBLASt matrix mul: "
                << std::fixed
                << Utils::measure([&cl_mx, &cl_my, &cl_mz]() {
                    cl_mx.dot(cl_my, cl_mz);
                    std::cout << "(" << cl_mz.sum() << ")" << " ";
                })
                << "s" << std::endl;

        delete[] data_x;
        delete[] data_y;
        delete[] data_z;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Exited" << std::endl;

    return EXIT_SUCCESS;
}