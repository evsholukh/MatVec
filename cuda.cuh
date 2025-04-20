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

        const int n = this->size();
        if (n == 1) {
            return this->head();
        }
        const int blockSize = 1024;
        const int blockCount = (n + blockSize - 1) / blockSize;

        T *input = this->data();
        T *d_input = nullptr,
          *d_partial = nullptr;

        Matrix<T> out_mat = Matrix<T>::zeros(1, blockCount);
        T* h_partial = out_mat.data();

        CHECK_CUDA(cudaMalloc(&d_input, n*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_partial, blockCount*sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_input, input, n*sizeof(T), cudaMemcpyHostToDevice));

        reduceSumKernel<<<blockCount, blockSize, blockSize*sizeof(T)>>>(d_input, d_partial, n);
        CHECK_CUDA(cudaGetLastError());
        // CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_partial, d_partial, blockCount*sizeof(T), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_partial));

        // out_mat.show();

        return MatrixCuda(out_mat).sum();
    }

    Matrix<T> add(Matrix<T> &o) override {
        MatrixCuda tmp(*this);
        return MatrixCuda(o).add(tmp);
    }

    MatrixCuda<T> add(MatrixCuda<T> &o) {

        const int n = this->size();
        const int blockSize = 1024;
        const int gridSize = (n + blockSize - 1) / blockSize;
        
        T *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;

        std::vector<T> res(n);

        T *h_x = this->data();
        T *h_y = o.data();
        T *h_z = res.data();

        CHECK_CUDA(cudaMalloc(&d_x, n*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_y, n*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_z, n*sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, n*sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, h_y, n*sizeof(T), cudaMemcpyHostToDevice));

        vectorAddKernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, n);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(h_z, d_z, n*sizeof(T), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));
        CHECK_CUDA(cudaFree(d_z));

        Matrix<T> tmp(res, this->N, this->M);
        return MatrixCuda<T>(tmp);
    }

    Matrix<T> dot(Matrix<T> &o) override {
        MatrixCuda tmp(*this);
        return MatrixCuda(o).dot(tmp);
    }

    MatrixCuda<T> dot(MatrixCuda<T> &o) {
        T *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;

        const int n = this->size();
        const int m = o.size();
        const int k = this->M*o.N;

        T *h_x = this->data();
        T *h_y = o.data();

        CHECK_CUDA(cudaMalloc(&d_x, n*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_y, m*sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_z, k*sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, n*sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, h_y, m*sizeof(T), cudaMemcpyHostToDevice));

        const int blockSize = 32;

        dim3 block(blockSize, blockSize); // 1024
        dim3 grid((o.N + blockSize - 1)/blockSize, (this->M + blockSize - 1)/blockSize);

        matrixMulKernel<<<grid, block>>>(d_x, d_y, d_z, this->M, this->N, o.N);
        CHECK_CUDA(cudaGetLastError());

        std::vector<T> res(k);
        T *h_z = res.data();

        CHECK_CUDA(cudaMemcpy(h_z, d_z, k*sizeof(T), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));
        CHECK_CUDA(cudaFree(d_z));

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
