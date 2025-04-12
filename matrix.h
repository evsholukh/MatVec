#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <algorithm> 


template <typename T>
class Matrix {

protected:
    std::vector<T> data;
    size_t N, M;

public:
    Matrix(std::vector<T> &data, size_t N, size_t M) : data(data), N(N), M(M) {
        if (N*M != data.size()) {
            throw std::runtime_error("Invalid matrix size");
        }
    }

    static Matrix zeros(size_t N, size_t M) {
        std::vector<T> vec(N*M);

        return Matrix(vec, N, M);
    }

    static Matrix random(const size_t N, const size_t M) {
        size_t size = N*M;
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> dist(-1, 1);
        std::vector<T> vec(size);

        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(generator);
        }
        return Matrix(vec, N, M);
    }

    void reshape(const size_t N, const size_t M) {
        if (N*M != this.data.size()) {
            throw std::runtime_error("Invalid shape");
        }
        this->N = N;
        this->M = M;
    }

    Matrix transposed() {
        return Matrix(this->data, this->M, this->N);
    }

    Matrix zeros_like() {
        return Matrix::zeros(this->N, this->M);
    }

    int size_mb() {
        return (sizeof(T) * data.size()) / (1024 * 1024);
    }

    virtual T sum() {
        return std::accumulate(data.begin(), data.end(), T(0));
    }

    virtual Matrix add(Matrix &o) {
        if (data.size() != o.data.size()) {
            throw std::runtime_error("Invalid size");
        }
        Matrix dst = this->zeros_like();

        std::transform(this->data.begin(), this->data.end(), o.data.begin(), dst.data.begin(), std::plus<T>());
        return dst;
    }

    virtual Matrix<T> dot(Matrix<T> &o) {
        if (this->M != o.N) {
            throw std::runtime_error("Invalid size");
        }
        Matrix dst = Matrix::zeros(this->N, o.M);

        for (size_t i = 0; i < this->N; i++) {
            for (size_t j = 0; j < o.M; j++) {

                for (size_t k = 0; k < this->M; k++) {
                    dst.data[dst.M*i + j] += this->data[this->M*i + k] * o.data[o.M*k + j];
                }
            }
        }
        return dst;
    }

    T mse(Matrix &o) {
        T res = 0.0f;
        for (size_t i = 0; i < data.size(); i++) {
            res += pow(data[i] - o.data[i], 2);
        }
        return res;
    }

    void print() {
        std::cout << "[";
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                std::cout << data[i*N+j];

                if (!(i == N - 1 && j == M - 1)) {
                    std::cout << ", ";
                }
            }
            if (i != N - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << "]"<< std::endl;
    }
};
