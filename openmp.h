#pragma once

#include "vector.h"
#include "matrix.h"


template <typename T = void>
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

    static std::string getOpenMPVersion() {
        #ifndef _OPENMP
            #define _OPENMP 0
        #endif
        std::unordered_map<unsigned,std::string> map{
            {199810,"1.0"},
            {200203,"2.0"},
            {200505,"2.5"},
            {200805,"3.0"},
            {201107,"3.1"},
            {201307,"4.0"},
            {201511,"4.5"},
            {201811,"5.0"},
            {202011,"5.1"},
            {202111,"5.2"},
            {202411,"6.0"},
            {0, ""}
        };
        return map.at(_OPENMP);
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