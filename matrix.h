#pragma once

#include "vector.h"


template <typename T>
class Matrix : public Vector<T> {

public:
    Matrix(T *data, size_t cols, size_t rows) : Vector<T>(data, cols*rows),
        _cols(cols), _rows(rows) { }

    size_t cols() const { return _cols; }

    size_t rows() const { return _rows; }

    Vector<T> row(size_t n) const { return Vector<T>(this->data()+_cols*n, _cols); }

    Vector<T> col(size_t n) const {
        T *h_c = new T[_rows]; // [!]
        T *data = this->data();
        for (size_t i = 0; i < _rows; i++) {
            h_c[i] = data[i*_rows + n];
        }
        return Vector<T>(h_c, _rows);
    }

    virtual void dot(const Matrix<T> &o, Matrix<T> &r) const {
        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < o._cols; j++) {
                Vector<T> vcol = o.col(j);
                Vector<T> vrow = this->row(i);

                r._data[o._cols*i + j] = vrow.dot(vcol);
                delete[] vcol.data();
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