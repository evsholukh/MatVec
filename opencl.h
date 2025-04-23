#pragma once

#include "matrix.h"


class VectorOpenCL : public Vector<float> {

public:
    VectorOpenCL(Vector<float> vec) : Vector<float>(vec) { }

    float dot(const Vector<float> &o) const override {
        return 0;
    }
};

class MatrixOpenCL : public Matrix<float> {

public:
    MatrixOpenCL(Matrix<float> mat) : Matrix<float>(mat) { }

    Matrix<float> dot(const Matrix<float> &o) const {
        return o;
    }
};
