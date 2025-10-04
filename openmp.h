#pragma once

#include "vector.h"
#include "matrix.h"


template <typename T>
class VectorOpenMP : public Vector<T> {

public: 
    VectorOpenMP(Vector<T> vec) : Vector<T>(vec) {}

    T dot(const Vector<T> &o) const override {
        T result = T(0.0);

        auto x = this->data();
        auto y = o.data();

        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < this->size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }
};

template <typename T>
class MatrixOpenMP : public Matrix<T> {

public:
    MatrixOpenMP(Matrix<T> mat) : Matrix<T>(mat) {}

    void dot(const Matrix<T> &o, Matrix<T> &r) const {

        auto C = r.data();
        auto A = this->data();
        auto B = o.data();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < this->rows(); ++i)
        for (int j = 0; j < o.cols(); ++j) {
            T sum = 0;

            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < this->cols(); ++k)
                sum += A[i * this->cols() + k] * B[k * o.cols() + j];

            C[i * o.cols() + j] = sum;
        }
    }
};