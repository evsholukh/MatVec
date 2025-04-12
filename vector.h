#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include "matrix.h"

template <typename T>
class Vector : public Matrix<T> {
public:
    Vector(std::vector<T> &data): Matrix<T>(data, data.size(), 1) {}
    Vector(Matrix<T> mat) : Matrix<T>(mat) {}

    static Vector random(const size_t size) {
        return Matrix<T>::random(size, 1);
    }
    Matrix<T> dot(Matrix<T> &o) override {
        auto tr = o.transposed();
        return Matrix<T>::dot(tr);
    }
};
