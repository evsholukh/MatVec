#pragma once

#include <iostream>


template <typename T>
class Vector {

public:
    Vector(T *data, size_t size): _data(data), _size(size) {}

    T* data() const {
        return _data;
    }

    size_t size() const {
        return _size;
    }

    virtual T sum() const {
        T val = _data[0];
        for (size_t i = 1; i < _size; i++) {
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
        return (*this * o).sum();
    }

    virtual Vector<T> operator+(const Vector<T> &o) const {
        T *new_data = new T(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] + o._data[i];
        }
        return Vector<T>(new_data, _size);
    }

    virtual Vector<T> operator*(const Vector<T> &o) const {
        T *new_data = new T(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] * o._data[i];
        }
        return Vector<T>(new_data, _size);
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

protected:
    T *_data;
    int _size;
};
