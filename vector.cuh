#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matrix.h"


class CUDA {

public:
    static std::string getDeviceName(const size_t idx = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, idx);

        return prop.name;
    }
};

template<typename T>
class VectorCuBLAS : public Vector<T> {

protected:
    T *d_x;

public:
    VectorCuBLAS(Vector<T> vec) : Vector<T>(vec) {
        cudaMalloc(&d_x, vec.size()*sizeof(T));
        cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
    }

    ~VectorCuBLAS() {
        cudaFree(d_x);
    }

    T dot(const Vector<T> &o) const override {
        // [!]
        if (auto* ocl = static_cast<const VectorCuBLAS<T>*>(&o)) {
            return this->dot(*ocl);
        }
        return Vector<T>::dot(o);
    }
};

template <>
float VectorCuBLAS<float>::dot(const VectorCuBLAS<float> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float result = 0;
    cublasSdot(handle, this->size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}

template <>
double VectorCuBLAS<double>::dot(const VectorCuBLAS<double> &o) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double result = 0;
    cublasDdot(handle, this->size(), d_x, 1, o.d_x, 1, &result);
    cublasDestroy(handle);

    return result;
}


template<typename T>
class MatrixCuBLAS : public Matrix<T> {

protected:
    T *d_A;

public:
    MatrixCuBLAS(Matrix<T> mat) : Matrix<T>(mat) {
        cudaMalloc(&d_A, mat.size()*sizeof(T));
        cudaMemcpy(d_A, mat.data(), mat.size()*sizeof(T), cudaMemcpyHostToDevice);
    }

    ~MatrixCuBLAS() {
        cudaFree(d_A);
    }

    void gemm(const Matrix<T> &o, Matrix<T> &r) const override {
        // [!]
        if (auto* ocl = static_cast<const MatrixCuBLAS<T>*>(&o)) {
            if (auto* rcl = static_cast<MatrixCuBLAS<T>*>(&r)) {
                this->gemm(*ocl, *rcl);
                return;
            }
        }
        Matrix<T>::gemm(o, r);
    }
};

template <>
void MatrixCuBLAS<float>::gemm(const MatrixCuBLAS<float> &o, MatrixCuBLAS<float> &r) const {
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
        o.d_A,         // B
        this->cols(),  // ldb
        &beta,         // beta
        r.d_A,         // C
        this->rows()); // ldc

    cudaMemcpy(r.data(), r.d_A, r.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
}

template <>
void MatrixCuBLAS<double>::gemm(const MatrixCuBLAS<double> &o, MatrixCuBLAS<double> &r) const {
    const double alpha = 1.0;
    const double beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDgemm(
        handle,        // handle
        CUBLAS_OP_N,   // transa
        CUBLAS_OP_N,   // transb
        this->rows(),  // m
        o.cols(),      // n
        this->cols(),  // k
        &alpha,        // alpha
        d_A,           // A
        this->rows(),  // lda
        o.d_A,         // B
        this->cols(),  // ldb
        &beta,         // beta
        r.d_A,         // C
        this->rows()); // ldc

    cudaMemcpy(r.data(), r.d_A, r.size()*sizeof(double), cudaMemcpyDeviceToHost);
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
class VectorCUDA : public Vector<T> {

protected:
    T *d_x;
    size_t blockSize, gridSize;

public:
    VectorCUDA(Vector<T> &vec, size_t blockSize = 1024, size_t gridSize = 512) : Vector<T>(vec),
        blockSize(blockSize),
        gridSize(gridSize) {

        cudaMalloc(&d_x, vec.size()*sizeof(T));
        cudaMemcpy(d_x, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
    }

    ~VectorCUDA() {
        cudaFree(d_x);
    }

    T dot(const Vector<T> &o) const override {
        // [!]
        if (auto* ocl = static_cast<const VectorCUDA<T>*>(&o)) {
            return this->dot(*ocl);
        }
        return Vector<T>::dot(o);
    }

    T dot(const VectorCUDA<T> &o) const {
        const size_t sharedMemSize = blockSize * sizeof(T);

        T *d_r;
        cudaMalloc(&d_r, gridSize * sizeof(T));

        reduceDotKernel<T><<<gridSize, blockSize, sharedMemSize>>>(this->d_x, o.d_x, d_r, this->size());

        T *res_data = new T[gridSize];
        Vector<T> vec(res_data, gridSize);

        cudaMemcpy(res_data, d_r, gridSize * sizeof(T), cudaMemcpyDeviceToHost);

        auto res = vec.sum();
        delete[] res_data; // [!]

        cudaFree(d_r); // [!]

        return res;
    }
};
