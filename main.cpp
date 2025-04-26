#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

#include "matrix.h"
#include "vector.h"
#include "utils.h"



int main(int argc, char **argv) {

    size_t N, M;

    std::cout << "N: ";
    std::cin >> N;

    std::cout << "M: ";
    std::cin >> M;

    try {
        std::cout << "Randomization array (size: " << N*M << ").." << std::endl;

        float *data_x = random_vector<float>(N*M);
        float *data_y = random_vector<float>(N*M);
        float *data_z = random_vector<float>(N*N);

        Vector<float> vx(data_x, N*M);
        Vector<float> vy(data_x, N*M);

        Matrix<float> mx(data_x, N, M);
        Matrix<float> my(data_y, M, N);
        Matrix<float> mz(data_z, N, N);

        std::cout << "Memory size: " << mx.size_mb() << "MB" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "C++ vector dot: "
                  << std::fixed
                  << Utils::measure([&vx, &vy]() {
                      std::cout << "(" << vx.dot(vy) << ")" << " ";
                  })
                  << "s" << std::endl;

        std::cout << std::left 
                  << std::setw(20)
                  << "C++ matrix mul: "
                  << std::fixed
                  << Utils::measure([&mx, &my, &mz]() {
                        mx.dot(my, mz);
                        std::cout << "(" << mz.sum() << ")" << " ";
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