#pragma once

#include "measured.h"
#include "matrix.h"
#include "vector.h"


template <typename T>
class MatrixAdd : public TimeMeasured {
protected:
    Matrix<T> &x, &y;
public:
    MatrixAdd(Matrix<T> &x, Matrix<T> &y): x(x), y(y) {}

    void perform() override {
        Matrix z = this->x.add(this->y);
        std::cout << z.sum() << " ";
    }
};

template <typename T>
class MatrixSum : public TimeMeasured {
protected:
    Matrix<T> &x;
public:
    MatrixSum(Matrix<T> &x): x(x) {}

    void perform() override {
        std::cout << this->x.sum() << " ";
    }
};

template <typename T>
class MatrixMul : public TimeMeasured {
protected:
    Matrix<T> &x, &y;
public:
    MatrixMul(Matrix<T> &x, Matrix<T> &y) : x(x), y(y) {}

    void perform() override {
        Matrix z  = this->x.dot(this->y);
        std::cout << z.sum() << " ";
    }
};

template <typename T>
class VectorAdd : public TimeMeasured {
protected:
    Vector<T> &x, &y;
public:
    VectorAdd(Vector<T> &x, Vector<T> &y): x(x), y(y) {}

    void perform() override {
        Vector z = this->x.add(this->y);
        std::cout << z.sum() << " ";
    }
};

template <typename T>
class VectorSum : public TimeMeasured {
protected:
    Vector<T> &x;
public:
    VectorSum(Vector<T> &x): x(x) {}

    void perform() override {
        std::cout << this->x.sum() << " ";
    }
};

template <typename T>
class VectorMul : public TimeMeasured {
protected:
    Vector<T> &x, &y;
public:
    VectorMul(Vector<T> &x, Vector<T> &y) : x(x), y(y) {}

    void perform() override {
        Vector z  = this->x.dot(this->y);
        std::cout << z.sum() << " ";
    }
};
