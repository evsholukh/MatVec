#pragma once

#include "vector.h"


template <typename T>
class VectorOpenMP : public Vector<T> {

public: 
    VectorOpenMP(Vector<T> vec) : Vector<T>(vec) {}

    T dot(const Vector<T> &o) const override {
        T result = T(0.0);

        auto x = this->data();
        auto y = o.data();

        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < this->size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }
};
