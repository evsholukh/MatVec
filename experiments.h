#pragma once

#include "measured.h"
#include "matrix.h"
#include "vector.h"


template <typename T>
class MatrixAdd : public TimeMeasured {
protected:
    const Matrix<T> _x, _y;
public:
    MatrixAdd(const Matrix<T> x, const Matrix<T> y): _x(x), _y(y) {}

    void perform() override {
        std::cout << (_x + _y).sum() << " ";
    }
};

template <typename T>
class MatrixSum : public TimeMeasured {
protected:
    const Matrix<T> _x;
public:
    MatrixSum(const Matrix<T> x): _x(x) {}

    void perform() override {
        std::cout << _x.sum() << " ";
    }
};

template <typename T>
class MatrixMul : public TimeMeasured {
protected:
    const Matrix<T> _x, _y;
public:
    MatrixMul(const Matrix<T> x, const Matrix<T> y) : _x(x), _y(y) {}

    void perform() override {
        std::cout << (_x * _y).sum() << " ";
    }
};

