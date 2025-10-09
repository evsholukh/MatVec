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
    static std::string getDeviceName(const size_t idx = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, idx);

        return prop.name;
    }
};

template<typename T>
class VectorCuBLAS {

protected:
    Vector<T> vec;
    T *d_x;

public:
    VectorCuBLAS(Vector<T> vec) : vec(vec) {
        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~VectorCuBLAS() {
        cudaFree(d_x);
    }

    T dot(const VectorCuBLAS<T> &o) const {
        return T(0);
    }
};

template <>
float VectorCuBLAS<float>::dot(const VectorCuBLAS<float> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float result = 0;
    cublasSdot(handle, vec.size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}

template <>
double VectorCuBLAS<double>::dot(const VectorCuBLAS<double> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double result = 0;
    cublasDdot(handle, vec.size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}


template<typename T>
class MatrixCuBLAS {

protected:
    Matrix<T> mat;
    T *d_A;

public:
    MatrixCuBLAS(Matrix<T> mat) : mat(mat) {
        CHECK_CUDA(cudaMalloc(&d_A, mat.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_A, mat.data(), mat.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~MatrixCuBLAS() {
        cudaFree(d_A);
    }

    void dot(const MatrixCuBLAS<T> &o, MatrixCuBLAS<T> &r) const {}
};

template <>
void MatrixCuBLAS<float>::dot(const MatrixCuBLAS<float> &o, MatrixCuBLAS<float> &r) const {
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
void MatrixCuBLAS<double>::dot(const MatrixCuBLAS<double> &o, MatrixCuBLAS<double> &r) const {
    const double alpha = 1.0;
    const double beta = 0.0;

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

template<typename T>
__global__ void reduceDotKernel(const T* x, const T* y, T *r, int n) {
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    T sum = T(0.0);
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

template <typename T>
class VectorCUDA {

protected:
    Vector<T> vec;
    T *d_x;
    size_t blockSize, gridSize;

public:
    VectorCUDA(
        Vector<T> vec,
        size_t blockSize = 1024,
        size_t gridSize = 512
    ) : vec(vec),
        blockSize(blockSize),
        gridSize(gridSize) {

        CHECK_CUDA(cudaMalloc(&d_x, vec.size()*sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
    }

    ~VectorCUDA() {
        cudaFree(d_x);
    }

    T dot(const VectorCUDA<T> &o) const {
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
