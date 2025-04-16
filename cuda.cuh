#pragma once

#include "matrix.h"
#include "vector.h"

#include <cuda_runtime.h>


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


template <typename T>
class MatrixCuda : public Matrix<T> {

public:
    MatrixCuda(Matrix<T> &mat) : Matrix<T>(mat) {}

    T sum() override {
        T *d_input = nullptr, *d_partial = nullptr, *d_output = nullptr;
        T h_output = 0.0;

        const int n = this->data.size();
        T *input = this->data.data();

        CHECK_CUDA(cudaMalloc(&d_input, n*sizeof(T)));

        const int blockSize = 1024;
        const int gridSize = (n + blockSize - 1) / blockSize;

        CHECK_CUDA(cudaMalloc(&d_partial, gridSize*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_output, sizeof(T))); 

        CHECK_CUDA(cudaMemcpy(d_input, input, n*sizeof(T), cudaMemcpyHostToDevice));

        reduceSumKernel<<<gridSize, blockSize, blockSize*sizeof(T)>>>(d_input, d_partial, n);
        CHECK_CUDA(cudaGetLastError());
    
        reduceSumKernel<<<1, blockSize, blockSize*sizeof(T)>>>(d_partial, d_output, gridSize);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_partial));
        CHECK_CUDA(cudaFree(d_output));

        return h_output;
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
        const int BLOCK_SIZE = 512;
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


__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
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
