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
        CHECK_CUDA(cudaMalloc(&d_C, this->rows()*o.cols()*sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, this->data(), this->size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, o.data(), this->size()*sizeof(float), cudaMemcpyHostToDevice));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, o.cols(), this->rows(), this->cols(), &alpha, d_B, o.cols(), d_A, o.rows(), &beta, d_C, o.cols());

        CHECK_CUDA(cudaMemcpy(r.data(), d_C, this->rows()*o.cols()*sizeof(float), cudaMemcpyDeviceToHost));
        cublasDestroy(handle);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};
