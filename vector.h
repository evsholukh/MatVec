#pragma once

#include <iostream>


template <typename T>
class Vector {

public:
    Vector(T *arr, size_t n) : arr(arr), n(n) {}

    T *data() const { return arr; }

    size_t size() const { return n; }

    virtual T sum() const {
        T val = T(0);
        for (size_t i = 0; i < n; i++) {
            val += arr[i];
        }
        return val;
    }

    virtual void add(const Vector<T> &o) {
        for (size_t i = 0; i < n; i++) {
            arr[i] += o.arr[i];
        }
    }

    virtual void mul(const Vector<T> &o) {
        for (size_t i = 0; i < n; i++) {
            arr[i] *= o.arr[i];
        }
    }

    virtual T dot(const Vector<T> &o) const {
        T val = T(0);
        for (size_t i = 0; i < n; i++) {
            val += arr[i] * o.arr[i];
        }
        return val;
    }

    virtual void print() const {
        std::cout << "[";
        for (size_t i = 0; i < n; i++) {
            std::cout << arr[i];
            if (i != n - 1) {
                std::cout << ",";
            }
        }
        std::cout << "]";
    }

    virtual size_t size_mb() const { return (sizeof(T) * n) / (1024 * 1024); }

protected:
    T *arr;
    size_t n;
};


class VectorFloat : public Vector<float> {

public:
    VectorFloat(float *arr, size_t n) : Vector(arr, n) {}

    float dot(const VectorFloat &o) const {
        double val = 0.0f;
        for (size_t i = 0; i < n; i++) {
            val += arr[i] * o.arr[i];
        }
        return val;
    }
};