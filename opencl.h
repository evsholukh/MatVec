#pragma once

#include <iostream>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include "matrix.h"


std::string decodeError(cl_int err);

#define CHECK_OPENCL(call) { \
    cl_int localErr = call; \
    if (localErr != CL_SUCCESS) { \
        std::cerr << "OpenCL Error in " << __FILE__ << ":" << __LINE__ << ": " << decodeError(localErr) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

template <typename T>
class MatrixOpenCL : public Matrix<T> {

protected:
    cl::Platform platform;
    cl::Device device;

public:
    MatrixOpenCL(Matrix<T> &v) : Matrix<T>(v) {

        std::vector<cl::Platform> platforms;
        CHECK_OPENCL(cl::Platform::get(&platforms));
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms.");
        }
        platform = platforms.back();

        std::vector<cl::Device> devices;
        CHECK_OPENCL(platform.getDevices(CL_DEVICE_TYPE_GPU, &devices));
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices.");
        }
        device = devices.back();
    }

    T sum() override {
        if (this->data.size() == 1) {
            return this->data.back();
        }
        size_t group_size = get_group_size();
        size_t total_size = this->data.size();
        if (total_size % group_size != 0) {
            total_size = ((total_size / group_size) + 1) * group_size;
        }
        this->data.resize(total_size, 0.0f);
        size_t groups_count = total_size / group_size;

        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, source());

        CHECK_OPENCL(program.build());

        cl_int err = CL_SUCCESS;
        cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * total_size, this->data.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer red_buf(context, CL_MEM_WRITE_ONLY, sizeof(T) * groups_count, nullptr, &err);
        CHECK_OPENCL(err);

        cl::Kernel sum_kernel(program, "reductionSumKernel", &err);
        CHECK_OPENCL(err);

        CHECK_OPENCL(sum_kernel.setArg(0, buf));
        CHECK_OPENCL(sum_kernel.setArg(1, group_size * sizeof(T), nullptr));
        CHECK_OPENCL(sum_kernel.setArg(2, red_buf));
        CHECK_OPENCL(sum_kernel.setArg(3, static_cast<uint32_t>(total_size)));

        cl::NDRange globalRange(total_size);
        cl::NDRange groupRange(group_size);

        CHECK_OPENCL(queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, globalRange, groupRange));

        std::vector<T> red_vec(groups_count);
        CHECK_OPENCL(queue.enqueueReadBuffer(red_buf, CL_TRUE, 0, sizeof(T)*groups_count, red_vec.data()));

        Matrix<T> red_mat(red_vec, groups_count, 1);
        return MatrixOpenCL(red_mat).sum();
    }

    Matrix<T> add(Matrix<T> &o) override {
        MatrixOpenCL cl_o(o);
        return MatrixOpenCL(*this).add(cl_o);
    }

    MatrixOpenCL<T> add(MatrixOpenCL<T> &o) {
        Matrix<T> tmp = this->zeros_like();
        MatrixOpenCL<T> res(tmp);

        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, source());

        cl_int err = CL_SUCCESS;
        CHECK_OPENCL(program.build());

        cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * this->data.size(), this->data.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer o_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * o.data.size(), o.data.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer res_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * res.data.size(), res.data.data(), &err);
        CHECK_OPENCL(err);

        cl::Kernel add_kernel(program, "add", &err);
        CHECK_OPENCL(err);

        CHECK_OPENCL(add_kernel.setArg(0, buf));
        CHECK_OPENCL(add_kernel.setArg(1, o_buf));
        CHECK_OPENCL(add_kernel.setArg(2, res_buf));
        CHECK_OPENCL(add_kernel.setArg(3, static_cast<uint32_t>(this->data.size())));

        cl::NDRange global_range(this->data.size());

        CHECK_OPENCL(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_range));
        CHECK_OPENCL(queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, sizeof(T)*res.data.size(), res.data.data()));

        return res;
    }

    Matrix<T> dot(Matrix<T> &o) override {
        MatrixOpenCL cl_o(o);
        return MatrixOpenCL(*this).dot(cl_o);
    }

    MatrixOpenCL<T> dot(MatrixOpenCL<T> &o) {
        Matrix<T> tmp = this->zeros_like();
        MatrixOpenCL<T> res(tmp);

        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, source());

        cl_int err = CL_SUCCESS;
        CHECK_OPENCL(program.build());

        cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * this->data.size(), this->data.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer o_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * o.data.size(), o.data.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer res_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(T) * res.data.size(), res.data.data(), &err);
        CHECK_OPENCL(err);

        cl::Kernel matmul_kernel(program, "matmul", &err);
        CHECK_OPENCL(err);

        CHECK_OPENCL(matmul_kernel.setArg(0, buf));
        CHECK_OPENCL(matmul_kernel.setArg(1, o_buf));
        CHECK_OPENCL(matmul_kernel.setArg(2, res_buf));
        CHECK_OPENCL(matmul_kernel.setArg(3, static_cast<uint32_t>(this->M)));

        cl::NDRange globalSize(this->M, o.N);

        CHECK_OPENCL(queue.enqueueNDRangeKernel(matmul_kernel, cl::NullRange, globalSize));
        CHECK_OPENCL(queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, sizeof(T)*res.data.size(), res.data.data()));

        return res;
    }

    void show_info() {
        std::string platform_name;
        CHECK_OPENCL(platform.getInfo(CL_PLATFORM_NAME, &platform_name));
        std::cout << "Platform: " << platform_name << std::endl;

        std::string device_name;
        CHECK_OPENCL(device.getInfo(CL_DEVICE_NAME, &device_name));
        std::cout << "Device: " << device_name << std::endl;

        cl_ulong mem_size;
        CHECK_OPENCL(device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &mem_size));
        std::cout << "GPU memory available: " << mem_size / (1024 * 1024) << "MB" << std::endl;
    }

private:
    
    const std::string source();
    const size_t get_group_size();

    static const std::string floatSource, doubleSource;
};

template <>
const std::string MatrixOpenCL<float>::source() {
    return floatSource;
}

template <>
const std::string MatrixOpenCL<double>::source() {
    return doubleSource;
}

template <>
const size_t MatrixOpenCL<float>::get_group_size() {
    return 256;
}

template <>
const size_t MatrixOpenCL<double>::get_group_size() {
    return 128;
}

template<>
const std::string MatrixOpenCL<float>::floatSource = R"(
    __kernel void add(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint n)
    {
        const uint i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }

    __kernel void reductionSumKernel(
        __global const float* data,
        __local float* local_data,
        __global float* result,
        const uint n)
    {
        const uint gid = get_global_id(0);
        const uint lid = get_local_id(0);
        const uint group_size = get_local_size(0);

        local_data[lid] = (gid < n) ? data[gid] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = group_size >> 1; i > 0; i >>= 1) {
            if (lid < i) {
                local_data[lid] += local_data[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            result[get_group_id(0)] = local_data[0];
        }
    }

    __kernel void matmul(
        __global const float* a,
        __global const float* b,
        __global float* c,
        const int n)
    {
        const uint row = get_global_id(0);
        const uint col = get_global_id(1);

        if (row >= n || col >= n) return;
        
        float acc = 0.0f;
        for (int k = 0; k < n; k++) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }

    __kernel void dot(
        __global const float* a,
        __global const float* b,
        __local float* local_data,
        __global float* result,
        const uint n)
    {
        const uint gid = get_global_id(0);
        const uint lid = get_local_id(0);
        const uint group_size = get_local_size(0);

        local_data[lid] = (gid < n) ? a[gid] * b[gid] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = group_size >> 1; i > 0; i >>= 1) {
            if (lid < i) {
                local_data[lid] += local_data[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            result[get_group_id(0)] = local_data[0];
        }
    }
)";

template<>
const std::string MatrixOpenCL<double>::doubleSource = R"(
    __kernel void add(
        __global const double* a,
        __global const double* b,
        __global double *c,
        const uint n)
    {
        int i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }

    __kernel void sum(
        __global const double* data,
        __local double* local_data,
        __global double* result,
        const uint n)
    {
        const uint gid = get_global_id(0);
        const uint lid = get_local_id(0);
        const uint group_size = get_local_size(0);

        local_data[lid] = (gid < n) ? data[gid] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = group_size >> 1; i > 0; i >>= 1) {
            if (lid < i) {
                local_data[lid] += local_data[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            result[get_group_id(0)] = local_data[0];
        }
    }

    __kernel void matmul(
        __global const double* a,
        __global const double* b,
        __global double* c,
        const int n)
    {
        const uint row = get_global_id(0);
        const uint col = get_global_id(1);

        if (row >= n || col >= n) return;
        
        double acc = 0.0f;
        for (int k = 0; k < n; k++) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }

    __kernel void dot(
        __global const double* a,
        __global const double* b,
        __local double* local_data,
        __global double* result,
        const uint n) {

        const uint gid = get_global_id(0);
        const uint lid = get_local_id(0);
        const uint group_size = get_local_size(0);

        local_data[lid] = (gid < n) ? a[gid] * b[gid] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = group_size >> 1; i > 0; i >>= 1) {
            if (lid < i) {
                local_data[lid] += local_data[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            result[get_group_id(0)] = local_data[0];
        }
    }
)";


template <typename T>
class VectorOpenCL : public Vector<T> {

public:
    VectorOpenCL(Vector<T> &vec) : Vector<T>(vec) {}

    T sum() override {
        return MatrixOpenCL(*this).sum();
    }

    Matrix<T> add(Matrix<T> &o) override {
        MatrixOpenCL tmp(o);
        return MatrixOpenCL(*this).add(tmp);
    }

    Matrix<T> dot(Matrix<T> &o) override {
        MatrixOpenCL tmp(o);
        return MatrixOpenCL(*this).dot(tmp);
    }
};

std::string decodeError(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
        case CL_INVALID_SPEC_ID: return "CL_INVALID_SPEC_ID";
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        default: return "Unknown OpenCL error " + std::to_string(err);
    }
}