#pragma once

#include "vector.h"


template <typename T>
class Matrix : public Vector<T> {

public:
    Matrix(T *data, size_t rows, size_t cols) : Vector<T>(data, cols*rows),
        _cols(cols), _rows(rows) { }

    size_t cols() const { return _cols; }

    size_t rows() const { return _rows; }

    Vector<T> row(size_t n) const { return Vector<T>(this->data()+_cols*n, _cols); }

    virtual void dot(const Matrix<T> &o, Matrix<T> &r) const {

        for (int i = 0; i < _rows; ++i) {
            for (int j = 0; j < o._cols; ++j) {
                T sum = 0.0f;
                for (int k = 0; k < _cols; ++k) {
                    sum += this->data()[i * _cols + k] * o.data()[k * o._cols + j];
                }
                r.data()[i * o._cols + j] = sum;
            }
        }
    }

    void print() const override {
        std::cout << "[";
        for (size_t i = 0; i < _rows; i++) {
            this->row(i).print();
            if (i != _rows - 1) {
                std::cout << ",";
                std::cout << std::endl;
            }
        }
        std::cout << "]";
        std::cout << std::endl;
    }

protected:
    size_t _cols, _rows;
};