#pragma once

#include <iostream>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define __CL_ENABLE_EXCEPTIONS // Enable exceptions

#include <CL/opencl.hpp>
#include "vector.h"


template <typename T>
class VectorOpenCL : public Vector<T> {
private:
    cl::Platform platform;
    cl::Device device;

    cl::Context *context;
    cl::CommandQueue *queue;
    cl::Program *program;
    cl::Buffer *buf, *red_buf;
    cl::Kernel *sum_kernel, *add_kernel, *dot_kernel;

    size_t N; // Aligned size of vector
    size_t groups_count;

public:
    VectorOpenCL(Vector<T> &v) : Vector<T>(v) {

        // Getting platforms
        std::vector<cl::Platform> platforms;
        cl_int err = cl::Platform::get(&platforms);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Getting platforms error: " + decodeError(err));
        }
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }
        platform = platforms.back();

        // Getting devices
        std::vector<cl::Device> devices;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Getting devices error: " + decodeError(err));
        }
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found");
        }
        // Setting fields
        device = devices.back();
        context = new cl::Context(device);
        queue = new cl::CommandQueue(*context, device);
        program = new cl::Program(*context, source());

        // Building program
        err = program->build();
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Compilation error: " + decodeError(err));
        }
        // Group size
        size_t group_size = get_group_size();

        // Extend vector to group size
        N = this->vec.size();
        if (N % group_size != 0) {
            N = ((N / group_size) + 1) * group_size;
        }
        // Padding vector
        this->vec.resize(N, 0.0f);

        // Work groups number
        groups_count = N / group_size;

        // Creating buffers on device
        buf = new cl::Buffer(*context, CL_MEM_READ_ONLY, sizeof(T) * this->vec.size());
        red_buf = new cl::Buffer(*context, CL_MEM_WRITE_ONLY, sizeof(T) * groups_count);

        // Copying vector to device
        err = queue->enqueueWriteBuffer(*buf, CL_TRUE, 0, sizeof(T) * this->vec.size(), this->vec.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Copying vector to device error: "+ decodeError(err));
        }
        // Creating kernels
        sum_kernel = new cl::Kernel(*program, "vector_sum");
        add_kernel = new cl::Kernel(*program, "vector_add");
        dot_kernel = new cl::Kernel(*program, "vector_dot");
    }

    ~VectorOpenCL() {
        // Removing  objects
        delete buf;
        delete red_buf;
        delete context;
        delete queue;
        delete program;
        delete sum_kernel;
        delete add_kernel;
        delete dot_kernel;
    }

    T sum() override {
        // Setting kernel args
        cl_int err = sum_kernel->setArg(0, *buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 0 error: " + decodeError(err));
        }
        err = sum_kernel->setArg(1, sizeof(T) * get_group_size()); // Local memory
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 1 error: " + decodeError(err));
        }
        err = sum_kernel->setArg(2, *red_buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 2 error: " + decodeError(err));
        }
        err = sum_kernel->setArg(3, static_cast<int>(N));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 3 error: " + decodeError(err));
        }
        // Data range
        cl::NDRange globalSize(N);

        // Group size range
        cl::NDRange groupSize(get_group_size());

        // Running kernel
        err = this->queue->enqueueNDRangeKernel(*sum_kernel, cl::NullRange, globalSize, groupSize);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Enquening kernel error: " + this->decodeError(err));
        }
        // Result vector
        std::vector<T> reduction_vec(groups_count);

        // Reading result
        err = this->queue->enqueueReadBuffer(*red_buf, CL_TRUE, 0, sizeof(T)*groups_count, reduction_vec.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Copying value from device error: " + this->decodeError(err));
        }
        return Vector(reduction_vec).sum();
    }

    void add(Vector<T> &o) override {
        return add(dynamic_cast<VectorOpenCL&>(o));
    }

    void add(VectorOpenCL &o) {

        // Setting args
        cl_int err = add_kernel->setArg(0, *buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 0 error: " + this->decodeError(err));
        }
        err = add_kernel->setArg(1, *o.buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 1 error: " + this->decodeError(err));
        }
        err = add_kernel->setArg(2, this->vec.size());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 2 error: " + this->decodeError(err));
        }
        // Work items count
        cl::NDRange global_range(this->vec.size());
        
        // Run kernel
        err = this->queue->enqueueNDRangeKernel(*add_kernel, cl::NullRange, global_range, cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Enquening kernel error: " + this->decodeError(err));
        }
        // Copying vector to host
        err = this->queue->enqueueReadBuffer(*buf, CL_TRUE, 0, sizeof(T)*this->vec.size(), this->vec.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Copying vector from device error: " + this->decodeError(err));
        }
    }

    T dot(Vector<T> &o) override {
        return dot(dynamic_cast<VectorOpenCL&>(o));
    }

    T dot(VectorOpenCL &o) {

        // Setting args
        cl_int err = dot_kernel->setArg(0, *buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 0 error: " + this->decodeError(err));
        }
        err = dot_kernel->setArg(1, *o.buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 1 error: " + this->decodeError(err));
        }
        err = dot_kernel->setArg(2, sizeof(T) * get_group_size()); // Local memory
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 2 error: " + this->decodeError(err));
        }
        err = dot_kernel->setArg(3, *red_buf);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 3 error: " + this->decodeError(err));
        }
        err = dot_kernel->setArg(4, static_cast<int>(N));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Setting arg 4 error: " + this->decodeError(err));
        }
        // Work-items count
        cl::NDRange globalSize(N);

        // Work-group count
        cl::NDRange groupSize(get_group_size());

        // Running kernel
        err = this->queue->enqueueNDRangeKernel(*dot_kernel, cl::NullRange, globalSize, groupSize);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Enquening kernel error: " + this->decodeError(err));
        }
        // Result vector
        std::vector<T> reduction_vec(groups_count);

        // Reading results
        err = this->queue->enqueueReadBuffer(*red_buf, CL_TRUE, 0, sizeof(T)*groups_count, reduction_vec.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Copying value from device error: " + this->decodeError(err));
        }
        return Vector(reduction_vec).sum();
    }

    void print_info() {
        std::string platform_name;
        cl_int err = this->platform.getInfo(CL_PLATFORM_NAME, &platform_name);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Getting platform name error: " + this->decodeError(err));
        }
        std::cout << "Platform: " << platform_name << std::endl;
    
        std::string device_name;
        err = this->device.getInfo(CL_DEVICE_NAME, &device_name);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Getting device name error: " + this->decodeError(err));
        }
        std::cout << "Device: " << device_name << std::endl;
    
        cl_ulong mem_size;
        err = this->device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &mem_size);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Getting device mem size error: " + this->decodeError(err));
        }
        std::cout << "GPU memory available: " << mem_size / (1024 * 1024) << "MB" << std::endl;
    }

private:
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
    const std::string source();
    const size_t get_group_size();

    static const std::string floatSource, doubleSource;
};

template <>
const std::string VectorOpenCL<float>::source() {
    return floatSource;
}

template <>
const std::string VectorOpenCL<double>::source() {
    return doubleSource;
}

template <>
const size_t VectorOpenCL<float>::get_group_size() {
    return 256;
}

template <>
const size_t VectorOpenCL<double>::get_group_size() {
    return 128;
}

template<>
const std::string VectorOpenCL<float>::floatSource = R"(
    __kernel void vector_add(
            __global float* a,
            __global const float* b,
            const uint n) {

        int i = get_global_id(0);
        if (i < n) {
            a[i] = a[i] + b[i];
        }
    }

    __kernel void vector_sum(
        __global float* data,
        __local float* local_data,
        __global float* result,
        const uint n) {

        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint group_size = get_local_size(0);

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

    __kernel void vector_dot(
        __global float* a,
        __global float* b,
        __local float* local_data,
        __global float* result,
        const uint n) {

        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint group_size = get_local_size(0);

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
const std::string VectorOpenCL<double>::doubleSource = R"(
    __kernel void vector_add(
        __global double* a,
        __global const double* b,
        const uint n) {

        int i = get_global_id(0);
        if (i < n) {
            a[i] = a[i] + b[i];
        }
    }

    __kernel void vector_sum(
        __global double* data,
        __local double* local_data,
        __global double* result, const uint n) {

        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint group_size = get_local_size(0);

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

    __kernel void vector_dot(
        __global double* a,
        __global double* b,
        __local double* local_data,
        __global double* result,
        const uint n) {

        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint group_size = get_local_size(0);

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
