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

template <typename T>
class VectorCorrected : public Vector<T> {

public:
    VectorCorrected(Vector<T> vec) : Vector<T>(vec) {}

    T dot(const Vector<T> &o) const override {
        T sum = 0.0f;
        T c = 0.0f; // коррекция

        for (size_t i = 0; i < this->n; i++) {
            T y = this->arr[i] * o.data()[i] - c;
            T t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }
};