#pragma once

#include "measured.h"
#include "vector.h"


template <typename T>
class VectorAdd : public TimeMeasured {
protected:
    Vector<T> &x, &y;
public:
    VectorAdd(Vector<T> &x, Vector<T> &y): x(x), y(y) {}

    void perform() override {
        x.add(y);
    }
};

template <typename T>
class VectorSum : public TimeMeasured {
protected:
    Vector<T> &x;
public:
    VectorSum(Vector<T> &x): x(x) {}

    void perform() override {
        x.sum();
    }
};

template <typename T>
class VectorDot : public TimeMeasured {
protected:
    Vector<T> &x, &y;
public:
    VectorDot(Vector<T> &x, Vector<T> &y) : x(x), y(y) {}

    void perform() override {
        this->x.dot(this->y);
    }
};
