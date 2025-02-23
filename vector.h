#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <algorithm> 


template <typename T>
class Vector {

protected:
    std::vector<T> vec;

public:
    Vector(std::vector<T> &vec) : vec(vec) {}

    static Vector random(size_t size) {
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> dist(-1, 1);
        std::vector<T> vec(size);

        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(generator);
        }
        return Vector(vec);
    }

    int size_mb() {
        return (sizeof(T) * vec.size()) / (1024 * 1024);
    }

    virtual T sum() {
        return std::accumulate(vec.begin(), vec.end(), T(0));
    }

    virtual void add(Vector &o) {
        std::transform(vec.begin(), vec.end(), o.vec.begin(), vec.begin(), std::plus<T>());
    }

    virtual T dot(Vector &o) {
        return std::inner_product(vec.begin(), vec.end(), o.vec.begin(), T(0));
    }

    T mse(Vector &o) {
        T res = 0.0f;
        for (size_t i = 0; i < vec.size(); i++) {
            res += pow(vec[i] - o.vec[i], 2);
        }
        return res;
    }

    void print() {
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << vec[i];
            if (i < vec.size() - 2)  {
                std::cout << ",";
            } else {
                std::cout << std::endl;
            }
        }
    }
};
