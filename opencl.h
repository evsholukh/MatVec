#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast.h>

#include "matrix.h"

std::string decodeError(cl_int err);

#define CHECK_OPENCL(call) if (call != CL_SUCCESS) throw std::runtime_error(decodeError(call));

class OpenCL {

public:
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
};

class VectorOpenCL : public Vector<float> {

public:
    VectorOpenCL(Vector<float> vec) : Vector<float>(vec) { }

    float dot(const Vector<float> &o) const override {
        return 0;
    }
};

class MatrixOpenCL : public Matrix<float> {

public:
    MatrixOpenCL(Matrix<float> mat) : Matrix<float>(mat) { }

    void dot(const Matrix<float> &o, Matrix<float> &r) const {

        auto platform = OpenCL::defaultPlatform();
        auto device = OpenCL::defaultDevice(platform);
        auto context = cl::Context(device);
        auto queue = cl::CommandQueue(context, device);
        auto event = cl_event{nullptr};

        auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, this->size()*sizeof(float));
        auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, o.size()*sizeof(float));
        auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, r.size()*sizeof(float));

        CHECK_OPENCL(queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, this->size()*sizeof(float), this->data()));
        CHECK_OPENCL(queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, o.size()*sizeof(float), o.data()));

        auto queue_plain = queue();

        auto status = clblast::Gemm(
            clblast::Layout::kRowMajor, // layout
            clblast::Transpose::kNo,    // a_transpose
            clblast::Transpose::kNo,    // b_transpose
            o.cols(),     // m
            this->rows(), // n
            this->cols(), // k
            1.0f,         // alpha
            device_a(),   // a_buffer
            0,            // a_offset
            this->rows(), // a_ld
            device_b(),   // b_buffer
            0,            // b_offset
            o.cols(),     // b_ld
            0.0f,         // beta
            device_c(),   // c_buffer
            0,            // c_offset
            r.cols(),     // c_ld
            &queue_plain, // queue
            &event        // event
        );

        if (status == clblast::StatusCode::kSuccess) {
            CHECK_OPENCL(clWaitForEvents(1, &event));
            CHECK_OPENCL(clReleaseEvent(event));
        }
        CHECK_OPENCL(queue.enqueueReadBuffer(device_c, CL_TRUE, 0, r.size()*sizeof(float), r.data()));
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