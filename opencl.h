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


class VectorOpenCL : public Vector<float> {

public:
    VectorOpenCL(Vector<float> vec) : Vector<float>(vec) { }

    float sum() const override {
        const int n = _size;
        const int blockSize = 1024;
        const int blockCount = (n + blockSize - 1) / blockSize;

        if (blockCount == 1) {
            return Vector<float>::sum();
        }

        cl::Platform platform = VectorOpenCL::defaultPlatform();
        cl::Device device = VectorOpenCL::defaultDevice(platform);
    
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, source);

        CHECK_OPENCL(program.build());

        cl_int err = CL_SUCCESS;
        cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, n * sizeof(float), _data, &err);
        CHECK_OPENCL(err);

        cl::Buffer red(context, CL_MEM_WRITE_ONLY, blockCount * sizeof(float), nullptr, &err);
        CHECK_OPENCL(err);

        cl::Kernel sum_kernel(program, "kernelVectorSum", &err);
        CHECK_OPENCL(err);

        CHECK_OPENCL(sum_kernel.setArg(0, buf));
        CHECK_OPENCL(sum_kernel.setArg(1, blockSize * sizeof(float), nullptr));
        CHECK_OPENCL(sum_kernel.setArg(2, red));
        CHECK_OPENCL(sum_kernel.setArg(3, static_cast<uint32_t>(n)));

        cl::NDRange globalRange(n);
        cl::NDRange groupRange(blockSize);

        CHECK_OPENCL(queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, globalRange, groupRange));

        float *res = new float[blockCount];
        Vector<float> red_vec(res, blockCount);
        CHECK_OPENCL(queue.enqueueReadBuffer(red, CL_TRUE, 0, blockCount * sizeof(float), res));

        float total = VectorOpenCL(red_vec).sum();
        delete[] res;

        return total;
    }

    Vector<float> operator*(const Vector<float> &o) const override {
        float *new_data = new float[_size];
        Vector<float> res(new_data, _size);

        cl::Platform platform = VectorOpenCL::defaultPlatform();
        cl::Device device = VectorOpenCL::defaultDevice(platform);

        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, source);

        cl_int err = CL_SUCCESS;
        CHECK_OPENCL(program.build());

        cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * _size, _data, &err);
        CHECK_OPENCL(err);

        cl::Buffer o_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * o.size(), o.data(), &err);
        CHECK_OPENCL(err);

        cl::Buffer res_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * res.size(), res.data(), &err);
        CHECK_OPENCL(err);

        cl::Kernel add_kernel(program, "kernelVectorMul", &err);
        CHECK_OPENCL(err);

        CHECK_OPENCL(add_kernel.setArg(0, buf));
        CHECK_OPENCL(add_kernel.setArg(1, o_buf));
        CHECK_OPENCL(add_kernel.setArg(2, res_buf));
        CHECK_OPENCL(add_kernel.setArg(3, static_cast<uint32_t>(_size)));

        cl::NDRange global_range(_size);

        CHECK_OPENCL(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_range));
        CHECK_OPENCL(queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res.size()*sizeof(float), res.data()));

        return res;
    }

    float dot(const Vector<float> &o) const override {
        return VectorOpenCL((*this) * VectorOpenCL(o)).sum();
    }

    static cl::Platform defaultPlatform() {
        std::vector<cl::Platform> platforms;
        CHECK_OPENCL(cl::Platform::get(&platforms));
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms.");
        }
        return platforms.back();
    }

    static cl::Device defaultDevice(cl::Platform platform) {
        std::vector<cl::Device> devices;
        CHECK_OPENCL(platform.getDevices(CL_DEVICE_TYPE_GPU, &devices));
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices.");
        }
        return devices.back();
    }

    static int memoryAvailable(cl::Device device) {
        cl_ulong mem_size;
        CHECK_OPENCL(device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &mem_size));

        return mem_size / (1024 * 1024);
    }

public:
    static const std::string source;
};


class MatrixOpenCL : public VectorOpenCL, public Matrix<float> {

public:
    MatrixOpenCL(Matrix<float> mat) : VectorOpenCL(mat), Matrix<float>(mat), Vector<float>(mat) { }

    Matrix<float> dot(const Matrix<float> &o) const {
        float *new_data = new float[_rows*o.cols()];
        Matrix<float> mat(new_data, _rows, o.cols());

        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < o.cols(); j++) {
                Vector<float> c = o.col(j);
                Vector<float> r = this->row(i);

                new_data[o.cols()*i + j] = VectorOpenCL(r).dot(c);
                delete[] c.data();
            }
        }
        return mat;
    }

    using VectorOpenCL::sum;
    using VectorOpenCL::operator*;
};

const std::string VectorOpenCL::source = R"(
    __kernel void kernelVectorAdd(
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

    __kernel void kernelVectorMul(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint n)
    {
        const uint i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] * b[i];
        }
    }

    __kernel void kernelVectorSum(
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

    __kernel void kernelMatrixDot(
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
)";


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