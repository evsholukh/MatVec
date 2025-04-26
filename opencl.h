#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast.h>

#include "matrix.h"


class OpenCL {

public:
    static cl::Platform defaultPlatform() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms.");
        }
        return platforms.back();
    }

    static cl::Device defaultDevice(cl::Platform platform) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices.");
        }
        return devices.back();
    }

    static int memoryAvailable(cl::Device device) {
        cl_ulong mem_size;
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &mem_size);

        return mem_size / (1024 * 1024);
    }
};

class VectorOpenCL : public Vector<float> {

public:
    VectorOpenCL(Vector<float> vec) : Vector<float>(vec) { }

    float dot(const Vector<float> &o) const override {
        auto platform = OpenCL::defaultPlatform();
        auto device = OpenCL::defaultDevice(platform);
        auto context = cl::Context(device);
        auto queue = cl::CommandQueue(context, device);
        auto event = cl_event{nullptr};

        auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, this->size()*sizeof(float));
        auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, o.size()*sizeof(float));
        auto device_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float));

        queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, this->size()*sizeof(float), this->data());
        queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, o.size()*sizeof(float), o.data());

        auto queue_plain = queue();

        auto status = clblast::Dot<float>(
            this->size(), // size
            device_c(),   // result
            0,            // offset
            device_a(),   // x_buffer
            0,            // x_offset
            1,            // x_inc
            device_b(),   // y_buffer
            0,            // y_offset
            1,            // y_inc
            &queue_plain, // queue
            &event        // event
        );

        if (status == clblast::StatusCode::kSuccess) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }

        float result = 0.0f;
        queue.enqueueReadBuffer(device_c, CL_TRUE, 0, sizeof(float), &result);

        return result;
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

        queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, this->size()*sizeof(float), this->data());
        queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, o.size()*sizeof(float), o.data());

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
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        queue.enqueueReadBuffer(device_c, CL_TRUE, 0, r.size()*sizeof(float), r.data());
    }
};
