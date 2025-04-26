#pragma once

#include "vector.h"
#include "matrix.h"

#include <cblas.h>

class VectorBLAS : public Vector<float> {

public:
    VectorBLAS(Vector<float> vec) : Vector(vec) {}

    float dot(const Vector<float> &o) const override {
        return cblas_sdot(this->size(), this->data(), 1, o.data(), 1);
    }
};


class MatrixBLAS : public Matrix<float> {

public:
    MatrixBLAS(Matrix<float> mat) : Matrix<float>(mat) { }
    
    void dot(const Matrix<float> &o, Matrix<float> &r) const override {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            this->rows(),
            o.cols(),
            this->cols(),
            1.0f, // alpha
            this->data(),
            this->cols(),
            o.data(),
            o.cols(),
            0.0f, // beta
            r.data(),
            o.cols());
    }
};