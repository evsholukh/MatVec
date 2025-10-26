#pragma once

#include "vector.h"


template <typename T>
class Matrix : public Vector<T> {

public:
    Matrix(T *data, size_t rows, size_t cols) : Vector<T>(data, cols*rows),
        _m(rows), _k(cols) { }

    size_t cols() const { return _m; }

    size_t rows() const { return _k; }

    Vector<T> row(size_t n) const { return Vector<T>(this->data()+_m*n, _m); }

    virtual void gemm(const Matrix<T> &o, Matrix<T> &r) const {

        for (int i = 0; i < _k; ++i) {
            for (int j = 0; j < o._m; ++j) {
                T sum = 0.0f;
                for (int k = 0; k < _m; ++k) {
                    sum += this->data()[i * _m + k] * o.data()[k * o._m + j];
                }
                r.data()[i * o._m + j] = sum;
            }
        }
    }

    void print() const override {
        std::cout << "[";
        for (size_t i = 0; i < _k; i++) {
            this->row(i).print();
            if (i != _k - 1) {
                std::cout << ",";
                std::cout << std::endl;
            }
        }
        std::cout << "]";
        std::cout << std::endl;
    }

protected:
    size_t _m, _k;
};