#pragma once

#include "vector.h"
#include "matrix.h"
#include "utils.h"

#include <cblas.h>

template <typename T = void>
class VectorBLAS : public Vector<T> {

public:
    VectorBLAS(Vector<T> vec) : Vector<T>(vec) {}

    virtual T dot(const Vector<T> &o) const {
        return T(0);
    };

    static std::string getOpenBLASVersion() {
        std::string str(OPENBLAS_VERSION);
        Utils::ltrim(str);
        Utils::rtrim(str);

        return str;
    }
};

template <>
float VectorBLAS<float>::dot(const Vector<float> &o) const {
    return cblas_sdot(this->size(), this->data(), 1, o.data(), 1);
}

template <>
double VectorBLAS<double>::dot(const Vector<double> &o) const {
    return cblas_ddot(this->size(), this->data(), 1, o.data(), 1);
}

template <typename T>
class MatrixBLAS : public Matrix<T> {

public:
    MatrixBLAS(Matrix<T> mat) : Matrix<T>(mat) { }

    virtual void dot(const Matrix<T> &o, Matrix<T> &r) const {}
};


template <>
void MatrixBLAS<float>::dot(const Matrix<float> &o, Matrix<float> &r) const {
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

template <>
void MatrixBLAS<double>::dot(const Matrix<double> &o, Matrix<double> &r) const {
    cblas_dgemm(
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
