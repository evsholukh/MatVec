#pragma once

#include "vector.h"


template <typename T>
class Matrix : public Vector<T> {

public:
    Matrix(T *data, size_t cols, size_t rows) : Vector<T>(data, cols*rows), 
        _cols(cols), _rows(rows) { }

    size_t cols() const {
        return _cols;
    }

    size_t rows() const {
        return _rows;
    }

    Vector<T> row(size_t n) const {
        return Vector<T>(this->data()+_cols*n, _cols);
    }

    Vector<T> col(size_t n) const {
        T *new_data = new T(_rows); // [!]
        T *data = this->data();
        for (size_t i = 0; i < _rows; i++) {
            new_data[i] = data[i*_rows + n];
        }
        return Vector<T>(new_data, _rows);
    }

    virtual Matrix<T> dot(const Matrix<T> &o) const {
        T *new_data = new T(_rows*o._cols);
        Matrix<T> mat(new_data, _rows, o._cols);

        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < o._cols; j++) {
                Vector<T> c = o.col(j);
                Vector<T> r = this->row(i);

                new_data[o._cols*i + j] = r.dot(c);
                delete[] c.data();
            }
        }
        return mat;
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