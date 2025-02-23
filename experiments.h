#pragma once

#include "measured.h"
#include "vector.h"


template <typename T>
class VectorAdd : public TimeMeasured {
protected:
    Vector<T> x, y;
public:
    VectorAdd(Vector<T> x, Vector<T> y): x(x), y(y) {}

    void perform() override {
        this->x.add(y);
    }
};


