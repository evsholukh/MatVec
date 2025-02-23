#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <algorithm> 


template <typename T>
class Vector {

protected:
    std::vector<T> _vec;

public:
    Vector(std::vector<T> vec) : _vec(vec) {}
    Vector(size_t size) : Vector(Vector::random(size)) {}

    static Vector random(size_t size) {
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> dist(-1, 1);
        std::vector<T> vec(size);

        for (size_t i = 0; i < size; i++) {
            // vec[i] = dist(generator);
            vec[i] = 1.0;
        }
        return Vector(vec);
    }

    int size_mb() {
        return (sizeof(T) * _vec.size()) / (1024 * 1024);
    }

    virtual T sum() {
        return std::accumulate(_vec.begin(), _vec.end(), T(0));
    }

    virtual void add(Vector &o) {
        std::transform(_vec.begin(), _vec.end(), o._vec.begin(), _vec.begin(), std::plus<T>());
    }

    virtual T dot(Vector &o) {
        return std::inner_product(_vec.begin(), _vec.end(), o._vec.begin(), T(0));
    }

    T mse(Vector &o) {
        T res = 0.0f;
        for (size_t i = 0; i < _vec.size(); i++) {
            res += pow(_vec[i] - o._vec[i], 2);
        }
        return res;
    }

    void print() {
        for (size_t i = 0; i < _vec.size(); i++) {
            std::cout << _vec[i];
            if (i < _vec.size() - 2)  {
                std::cout << ",";
            } else {
                std::cout << std::endl;
            }
        }
    }

    // std::vector<T>& vec() {
    //     return _vec;
    // }
};
