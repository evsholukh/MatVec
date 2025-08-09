#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matrix.h"


#define CHECK_CUDA(val) handleError(val);

void handleError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in "
                  << __FILE__
                  << ":"
                  << __LINE__
                  << ": "
                  << cudaGetErrorString(err)
                  << std::endl;

        exit(EXIT_FAILURE);
    }
}

class VectorCuda : public Vector<float> {

public:
    VectorCuda(Vector<float> vec) : Vector<float>(vec) { }

    float dot(const Vector<float> &o) const override {
        float *d_x, *d_y;

        CHECK_CUDA(cudaMalloc(&d_x, this->size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, o.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, this->data(), this->size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, o.data(), this->size() * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        cublasCreate(&handle);

        float result = 0;
        cublasSdot(handle, this->size(), d_x, 1, d_y, 1, &result);

        cublasDestroy(handle);
        cudaFree(d_x);
        cudaFree(d_y);

        return result;
    }
};

class MatrixCuda : public Matrix<float> {

public:
    MatrixCuda(Matrix<float> mat) : Matrix<float>(mat) { }

    void dot(const Matrix<float> &o, Matrix<float> &r) const override {
        float *d_A, *d_B, *d_C;

        CHECK_CUDA(cudaMalloc(&d_A, this->size()*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, o.size()*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, r.size()*sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, this->data(), this->size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, o.data(), this->size()*sizeof(float), cudaMemcpyHostToDevice));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasSgemm(
            handle,        // handle
            CUBLAS_OP_N,   // transa
            CUBLAS_OP_N,   // transb
            this->rows(),  // m
            o.cols(),      // n
            this->cols(),  // k
            &alpha,        // alpha
            d_A,           // A
            this->rows(),  // lda
            d_B,           // B
            this->cols(),  // ldb
            &beta,         // beta
            d_C,           // C
            this->rows()); // ldc

        CHECK_CUDA(cudaMemcpy(r.data(), d_C, r.size()*sizeof(float), cudaMemcpyDeviceToHost));
        cublasDestroy(handle);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};


__global__ void reduceDotKernel(const float* x, const float* y, float *r, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += x[i] * y[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        r[blockIdx.x] = sdata[0];
    }
}

class VectorReduceCuda : public Vector<float> {

protected:
    size_t blockSize, gridSize;

public:
    VectorReduceCuda(Vector<float> vec, size_t blockSize = 1024, size_t gridSize = 512)
        : Vector<float>(vec), blockSize(blockSize), gridSize(gridSize) { }

    float dot(const Vector<float> &o) const override {
        float *d_x, *d_y, *d_r;

        CHECK_CUDA(cudaMalloc(&d_x, this->size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, o.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, this->data(), this->size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, o.data(), o.size() * sizeof(float), cudaMemcpyHostToDevice));

        const size_t sharedMemSize = blockSize * sizeof(float);

        CHECK_CUDA(cudaMalloc(&d_r, gridSize * sizeof(float)));

        reduceDotKernel<<<gridSize, blockSize, sharedMemSize>>>(d_x, d_y, d_r, this->size());

        float *res_data = new float[gridSize];
        Vector<float> vec(res_data, gridSize);

        cudaMemcpy(res_data, d_r, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        auto res = vec.sum();
        delete[] res_data;

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_r);

        return res;
    }
};
