
#include <iostream>
#include "vector.h"
#include "matrix.h"
#include "opencl.h"

int main(int argc, char** argv) {

    float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    Vector x(data, sizeof(data) / sizeof(float));
    // x.print();
    // x.add(x);
    // x.print();
    auto y = x + x;
    y.print();

    // auto z = y * x;
    // z.print();

    // z.mul(z);
    // z.print();

    std::cout << x.dot(x) << std::endl;

    Matrix mx(data, 3, 3);
    mx.print();
    mx.dot(mx).print();

    VectorOpenCL cl_x(x);

    std::cout << x.sum() << std::endl;
    std::cout << cl_x.sum() << std::endl;
    std::cout << (x * x).sum() << std::endl;
    std::cout << (cl_x * cl_x).sum() << std::endl;
    std::cout << x.dot(x) << std::endl;
    std::cout << cl_x.dot(cl_x) << std::endl;

    MatrixOpenCL cl_mx(mx);
    cl_mx.dot(mx).print();
    cl_mx.dot(cl_mx).print();
    cl_mx.dot2(cl_mx).print();
}