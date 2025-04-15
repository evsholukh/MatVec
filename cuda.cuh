#pragma once

#include "matrix.h"
#include "vector.h"

#include <cuda_runtime.h>


template <typename T>
class MatrixCuda : public Matrix<T> {

public:
    MatrixCuda(Matrix<T> &mat) : Matrix<T>(mat) {}

    T sum() override {
        return T(0);
    }

    Matrix<T> add(Matrix<T> &o) override {
        MatrixCuda tmp(*this);
        return MatrixCuda(o).add(tmp);
    }

    MatrixCuda<T> add(MatrixCuda<T> &o) {
        T *d_A, *d_B, *d_C;

        cudaError_t err = cudaMalloc(&d_A, this->data.size()*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc a error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_B, o.data.size()*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc b error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_C, this->data.size()*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc C error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(d_A, this->data.data(), this->data.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy A error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(d_B, o.data.data(), o.data.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy B error: " + std::string(cudaGetErrorString(err)));
        }
        const int BLOCK_SIZE = 1024;
        int num_blocks_1d = (this->data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        vectorAddKernel<<<num_blocks_1d, BLOCK_SIZE>>>(d_A, d_B, d_C, this->data.size());

        std::vector<T> res(this->data.size());

        err = cudaMemcpy(res.data(), d_C, res.size()*sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy C error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_A);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free A error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_B);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free B error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_C);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free C error: " + std::string(cudaGetErrorString(err)));
        }
        Matrix<T> tmp(res, this->N, this->M);
        return MatrixCuda<T>(tmp);
    }

    Matrix<T> dot(Matrix<T> &o) override {
        MatrixCuda tmp(*this);
        return MatrixCuda(o).dot(tmp);
    }

    MatrixCuda<T> dot(MatrixCuda<T> &o) {
        T *d_A, *d_B, *d_C;

        cudaError_t err = cudaMalloc(&d_A, this->data.size()*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc A error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_B, o.data.size()*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc B error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_C, this->M*o.N*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc C error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(d_A, this->data.data(), this->data.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy A error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(d_B, o.data.data(), o.data.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy B error: " + std::string(cudaGetErrorString(err)));
        }
        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((o.N + BLOCK_SIZE - 1)/BLOCK_SIZE, (this->M + BLOCK_SIZE - 1)/BLOCK_SIZE);

        matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, this->M, this->N, o.N);

        std::vector<T> res(this->M * o.N);
        err = cudaMemcpy(res.data(), d_C, res.size()*sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy C error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_A);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free A error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_B);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free B error: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaFree(d_C);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA free C error: " + std::string(cudaGetErrorString(err)));
        }
        Matrix<T> tmp(res, this->M, o.N);
        return MatrixCuda(tmp);
    }
};


__global__ void reductionSum(float *input, float *output, unsigned int n) {
    unsigned int block_size = blockDim.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int chunk_size = block_size * 2;
    unsigned int block_start = block_id * chunk_size;
    unsigned int left;  // holds index of left operand
    unsigned int right; // holds index or right operand
    unsigned int threads = block_size;

    for (unsigned int stride = 1; stride < chunk_size; stride *= 2, threads /= 2) {
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads && right < n) {
            input[left] += input[right];
        }
        __syncthreads();
    }
    if (!thread_id) {
        output[block_id] = input[block_start];
    }
}

__global__ void vectorAddKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void matrixMulKernel(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}
