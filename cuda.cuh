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

class CUDA {

public:
    static std::string deviceName(const size_t idx = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, idx);

        return prop.name;
    }
};

template<typename T>
class VectorCuda {

protected:
    Vector<T> vec;
    T *d_x;

public:
    VectorCuda(Vector<T> vec) : vec(vec) {
        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~VectorCuda() {
        cudaFree(d_x);
    }

    T dot(const VectorCuda<T> &o) const {
        return T(0);
    }
};

template <>
float VectorCuda::dot(const VectorCuda<float> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float result = 0;
    cublasSdot(handle, vec.size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}

template <>
double VectorCuda::dot(const VectorCuda<double> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double result = 0;
    cublasDdot(handle, vec.size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}


template<typename T>
class MatrixCuda {

protected:
    Matrix<T> mat;
    T *d_A;

public:
    MatrixCuda(Matrix<T> mat) : mat(mat) {
        CHECK_CUDA(cudaMalloc(&d_A, mat.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_A, mat.data(), mat.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~MatrixCuda() {
        cudaFree(d_A);
    }

    void dot(const MatrixCuda<T> &o, MatrixCuda<T> &r) const {}
};

template <>
void dot(const MatrixCuda<float> &o, MatrixCuda<float> &r) const {
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

template <>
void dot(const MatrixCuda<double> &o, MatrixCuda<double> &r) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDgemm(
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

    CHECK_CUDA(cudaMemcpy(r.mat.data(), r.d_A, r.mat.size()*sizeof(double), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
}


__global__ void reduceDotKernel(const T* x, const T* y, T *r, int n) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    T sum = 0.0f;
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
    Vector<T> vec;
    T *d_x;
    size_t blockSize, gridSize;

public:
    VectorReduceCuda(
        Vector<T> vec,
        size_t blockSize = 1024,
        size_t gridSize = 512
    ) : vec(vec),
        blockSize(blockSize),
        gridSize(gridSize) {

        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~VectorReduceCuda() {
        cudaFree(d_x);
    }

    T VectorReduceCuda::dot(const VectorReduceCuda<T> &o) const {
        const size_t sharedMemSize = blockSize * sizeof(T);

        T *d_r;
        CHECK_CUDA(cudaMalloc(&d_r, gridSize * sizeof(T)));

        reduceDotKernel<T><<<gridSize, blockSize, sharedMemSize>>>(d_x, o.d_x, d_r, vec.size());

        T *res_data = new T[gridSize];
        Vector<T> vec(res_data, gridSize);

        cudaMemcpy(res_data, d_r, gridSize * sizeof(T), cudaMemcpyDeviceToHost);

        auto res = vec.sum();
        delete[] res_data;

        cudaFree(d_r);

        return res;
    }
};
