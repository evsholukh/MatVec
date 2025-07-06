#pragma once

#include <iostream>


template <typename T>
class Vector {

public:
    Vector(T *data, size_t size): 
        _data(data), _size(size) {}

    T* data() const { return _data; }

    size_t size() const { return _size; }

    virtual T sum() const {
        T val = T(0);
        for (size_t i = 0; i < _size; i++) {
            val += _data[i];
        }
        return val;
    }

    virtual void add(const Vector<T> &o) {
        for (size_t i = 0; i < _size; i++) {
            _data[i] += o._data[i];
        }
    }

    virtual void mul(const Vector<T> &o) {
        for (size_t i = 0; i < _size; i++) {
            _data[i] *= o._data[i];
        }
    }

    virtual T dot(const Vector<T> &o) const {
        T val = T(0);

        for (size_t i = 0; i < _size; i++) {
            val += _data[i] * o._data[i];
        }
        return val;
    }

    virtual void print() const {
        std::cout << "[";
        for (size_t i = 0; i < _size; i++) {
            std::cout << _data[i];
            if (i != _size - 1) {
                std::cout << ",";
            }
        }
        std::cout << "]";
    }

    size_t size_mb() { return (sizeof(T) * _size) / (1024 * 1024); }

protected:
    T *_data;
    size_t _size;
};
