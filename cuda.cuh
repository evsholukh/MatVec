#pragma once

#include <cuda_runtime.h>
#include "matrix.h"


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


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

__global__ void vectorMulKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void matrixDotKernel(float *A, float *B, float *C, int m, int k, int n) {
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


class VectorCuda : virtual public Vector<float> {

public:
    VectorCuda(Vector<float> vec) : Vector<float>(vec) { }

    float sum() const override {
        const int n = _size;
        const int blockSize = 1024;
        const int gridSize = (n + blockSize - 1) / blockSize;

        if (gridSize == 1) {
            return Vector::sum();
        }
        float *d_input = nullptr, *d_partial = nullptr;
        float *h_partial = new float[gridSize];

        CHECK_CUDA(cudaMalloc(&d_input, n*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_partial, gridSize*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_input, _data, n*sizeof(float), cudaMemcpyHostToDevice));

        reduceSumKernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_input, d_partial, n);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(h_partial, d_partial, gridSize*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_partial));

        Vector<float> res(h_partial, gridSize);
        float sum = VectorCuda(res).sum();

        delete[] h_partial;
        return sum;
    }

    Vector<float> operator*(const Vector<float> &o) const override {
        const int n = _size;
        const int blockSize = 1024;
        const int gridSize = (n + blockSize - 1) / blockSize;
        
        float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
        float *h_z = new float[n];

        CHECK_CUDA(cudaMalloc(&d_x, n*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, n*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_z, n*sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, _data, n*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, o.data(), n*sizeof(float), cudaMemcpyHostToDevice));

        vectorMulKernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, n);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(h_z, d_z, n*sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));
        CHECK_CUDA(cudaFree(d_z));

        return Vector<float>(h_z, n);
    }

    float dot(const Vector<float> &o) const override {
        return VectorCuda((*this) * VectorCuda(o)).sum();
    }
};

class MatrixCuda : public VectorCuda, public Matrix<float> {

public:
    MatrixCuda(Matrix<float> mat) : VectorCuda(mat), Matrix<float>(mat), Vector<float>(mat) { }

    Matrix<float> dot(const Matrix<float> &o) const override {

        float *new_data = new float[_rows*o.cols()];
        Matrix<float> mat(new_data, _rows, o.cols());

        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < o.cols(); j++) {
                Vector<float> c = o.col(j);
                Vector<float> r = this->row(i);

                new_data[o.cols()*i + j] = VectorCuda(r).dot(c);
                delete[] c.data();
            }
        }
        return mat;
    }

    using VectorCuda::sum;
    using VectorCuda::operator*;
};
