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

class VectorCuda {

protected:
    Vector<float> vec;
    float *d_x;

public:
    VectorCuda(Vector<float> vec) : vec(vec) {
        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(float), cudaMemcpyHostToDevice));
    }

    ~VectorCuda() {
        cudaFree(d_x);
    }

    float dot(const VectorCuda &o) const {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float result = 0;
        cublasSdot(handle, vec.size(), d_x, 1, o.d_x, 1, &result);
        cublasDestroy(handle);

        return result;
    }
};

class MatrixCuda {

protected:
    Matrix<float> mat;
    float *d_A;

public:
    MatrixCuda(Matrix<float> mat) : mat(mat) {
        CHECK_CUDA(cudaMalloc(&d_A, mat.size()*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, mat.data(), mat.size()*sizeof(float), cudaMemcpyHostToDevice));
    }

    ~MatrixCuda() {
        cudaFree(d_A);
    }

    void dot(const MatrixCuda &o, MatrixCuda &r) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasSgemm(
            handle,        // handle
            CUBLAS_OP_N,   // transa
            CUBLAS_OP_N,   // transb
            mat.rows(),    // m
            o.mat.cols(),  // n
            mat.cols(),    // k
            &alpha,        // alpha
            d_A,           // A
            mat.rows(),    // lda
            o.d_A,         // B
            mat.cols(),    // ldb
            &beta,         // beta
            r.d_A,         // C
            mat.rows());   // ldc

        CHECK_CUDA(cudaMemcpy(r.mat.data(), r.d_A, r.mat.size()*sizeof(float), cudaMemcpyDeviceToHost));
        cublasDestroy(handle);
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

class VectorReduceCuda {

protected:
    Vector<float> vec;
    float *d_x;
    size_t blockSize, gridSize;

public:
    VectorReduceCuda(
        Vector<float> vec,
        size_t blockSize = 1024,
        size_t gridSize = 512
    ) : vec(vec),
        blockSize(blockSize),
        gridSize(gridSize) {

        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(float), cudaMemcpyHostToDevice));
    }

    ~VectorReduceCuda() {
        cudaFree(d_x);
    }

    float dot(const VectorReduceCuda &o) const {
        const size_t sharedMemSize = blockSize * sizeof(float);

        float *d_r;
        CHECK_CUDA(cudaMalloc(&d_r, gridSize * sizeof(float)));

        reduceDotKernel<<<gridSize, blockSize, sharedMemSize>>>(d_x, o.d_x, d_r, vec.size());

        float *res_data = new float[gridSize];
        Vector<float> vec(res_data, gridSize);

        cudaMemcpy(res_data, d_r, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        auto res = vec.sum();
        delete[] res_data;

        cudaFree(d_r);

        return res;
    }
};
