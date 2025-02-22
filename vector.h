#pragma once

#include <vector>
#include <random>


class Vector {

protected:
    std::vector<float> _vec;

public:
    Vector(std::vector<float> vec) : _vec(vec) {}
    Vector(size_t size) : Vector(Vector::random(size)) {}

    static Vector random(size_t size) {
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> dist(-1, 1);
        std::vector<float> vec(size);
        for (size_t i = 0; i < size; i++) {
            // vec[i] = dist(generator);
            vec[i] = 1.0;
        }
        return Vector(vec);
    }

    int size_mb() {
        return (sizeof(float) * _vec.size()) / (1024 * 1024);
    }

    virtual float sum() {
        float res = 0.0f;
        for (size_t i = 0; i < _vec.size(); i++) {
            res += _vec.at(i);
        }
        return res;
    }

    virtual void add(Vector &o) {
        for (size_t i = 0; i < _vec.size(); i++) {
            _vec[i] += o._vec[i];
        }
    }

    virtual float dot(Vector &o) {
        float res = 0.0f;
        for (size_t i = 0; i < _vec.size(); i++) {
            res += _vec[i] * o._vec[i];
        }
        return res;
    }

    float mse(Vector &o) {
        float res = 0.0f;
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

    std::vector<float>& vec() {
        return _vec;
    }
};
