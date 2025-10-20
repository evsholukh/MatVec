#pragma once

#include <chrono>
#include <random>

#include "vector.h"
#include "utils.h"


template<typename T>
class Bench {
public:
    virtual T perform() = 0;
};

class Metric {
    const unsigned long int ops;
    const double ms;

public:
    Metric(const unsigned long int ops, const double ms): ops(ops), ms(ms) {}

    double flops() const { return ops / ms; }

    double gflops() const { return ops / (ms * 1000000000.0); }
};

template<typename T>
class MetricResult : public Metric {

private:
    const T fResult;

public:
    MetricResult(const T result, const Metric &metric) : fResult(result),
        Metric(metric) { }

    MetricResult(const T result, const int ops, const double ms):
        fResult(result), Metric(ops, ms) { }

    T result() const {
        return fResult;
    }
};

template<typename T>
class DotFlops : public Bench<MetricResult<T>> {

private:
    const Vector<T> &x, &y;

public:
    DotFlops(const Vector<T> &x, const Vector<T> &y) : x(x), y(y) {}

    MetricResult<T> perform() override {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = x.dot(y);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> elapsed = end_time - start_time;

        auto ms = elapsed.count();
        auto ops = x.size() + x.size() - 1;
        auto metric = Metric(ops, ms);

        return MetricResult(result, metric);
    }
};


template<typename T>
class GEMMFlops : public Bench<MetricResult<T>> {

private:
    const Matrix<T> &x, &y;
    Matrix<T> &z;

public:
    GEMMFlops(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &z) : x(x), y(y), z(z) {}

    MetricResult<T> perform() override {
        auto start_time = std::chrono::high_resolution_clock::now();
        x.gemm(y, z);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> elapsed = end_time - start_time;

        auto ms = elapsed.count();
        auto ops = x.rows() * y.cols() * (2 * x.cols());
        auto metric = Metric(ops, ms);
        auto result = z.sum();

        return MetricResult(result, metric);
    }
};