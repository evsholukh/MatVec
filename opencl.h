#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast_c.h>

#include "matrix.h"


class OpenCL {

public:

    static cl::Context defaultContext() {
        auto device = defaultDevice();

        return cl::Context(device);
    }

    static cl::Platform defaultPlatform() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms.");
        }
        return platforms.back();
    }

    static cl::Device defaultDevice() {
        auto platform = defaultPlatform();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices.");
        }
        return devices.back();
    }

    static size_t maxGroupSize(const cl::Device &device) {
        size_t size;
        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);

        return size;
    }

    static int memoryAvailable(const cl::Device &device) {
        cl_ulong mem_size;
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &mem_size);

        return mem_size / (1024 * 1024);
    }
};


class VectorOpenCL {

public:
    static cl::Device defaultDevice;
    static cl::Context defaultContext;
    static cl::CommandQueue defaultQueue;

protected:
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _queue;
    cl::Program _program;

    Vector<float> _vec;
    cl::Buffer _deviceVec;

    size_t _blockSize, _blocksCount, _globalSize;

public:
    VectorOpenCL(
        Vector<float> vec,
        size_t blockSize = 0,
        cl::Device device = VectorOpenCL::defaultDevice,
        cl::Context context = VectorOpenCL::defaultContext,
        cl::CommandQueue queue = VectorOpenCL::defaultQueue
    ) : _vec(vec),
        _device(device),
        _context(context),
        _queue(queue) {

        if (blockSize == 0) {
            blockSize = OpenCL::maxGroupSize(device);
        }
        _blockSize = blockSize;
        _blocksCount = (vec.size() + blockSize - 1) / blockSize;
        _globalSize = _blockSize * _blocksCount;

        _deviceVec = cl::Buffer(_context, CL_MEM_READ_ONLY, _globalSize*sizeof(float));

        _program = cl::Program(_context, kernel);
        _program.build();
    }

    float dot(const VectorOpenCL &o) const {
        auto deviceResult = cl::Buffer(_context, CL_MEM_WRITE_ONLY, _blocksCount*sizeof(float));

        _queue.enqueueWriteBuffer(_deviceVec, CL_TRUE, 0, this->_vec.size()*sizeof(float), this->_vec.data());
        _queue.enqueueWriteBuffer(o._deviceVec, CL_TRUE, 0, o._vec.size()*sizeof(float), o._vec.data());

        auto addKernel = cl::Kernel(_program, "float_dot_prod");

        addKernel.setArg(0, _deviceVec);
        addKernel.setArg(1, o._deviceVec);
        addKernel.setArg(2, deviceResult);
        addKernel.setArg(3, _blockSize * sizeof(float), nullptr);

        auto globalRange = cl::NDRange(_globalSize);
        auto groupRange = cl::NDRange(_blockSize);

        float *resData = new float[_blocksCount];

        _queue.enqueueNDRangeKernel(addKernel, cl::NullRange, globalRange, groupRange);
        _queue.enqueueReadBuffer(deviceResult, CL_TRUE, 0, _blocksCount * sizeof(float), resData);

        auto resVec = Vector<float>(resData, _blocksCount);
        auto result = resVec.sum();

        delete[] resData;

        return result;
    }

    const std::string kernel = R"(
        __kernel void float_dot_prod(__global const float* a, __global const float* b,
            __global float* result, __local float* local_sum)
        {
            uint gid = get_global_id(0);
            uint lid = get_local_id(0);
            uint l_size = get_local_size(0);

            local_sum[lid] = a[gid] * b[gid];
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = l_size >> 1; i > 0; i >>= 1)
            {
                if (lid < i)
                    local_sum[lid] += local_sum[lid+i];
                barrier(CLK_LOCAL_MEM_FENCE);
            } 
            
            if (lid == 0)
                result[get_group_id(0)] = local_sum[0];
        }
    )";
};


cl::Device VectorOpenCL::defaultDevice = OpenCL::defaultDevice();

cl::Context VectorOpenCL::defaultContext = OpenCL::defaultContext();

cl::CommandQueue VectorOpenCL::defaultQueue = cl::CommandQueue(VectorOpenCL::defaultContext, VectorOpenCL::defaultDevice);


class VectorCLBlast : public VectorOpenCL {

public:

    VectorCLBlast(Vector<float> vec) : VectorOpenCL(vec) {}

    float dot(const VectorCLBlast &o) const {
        auto event = cl_event{nullptr};
        auto device_c = cl::Buffer(_context, CL_MEM_WRITE_ONLY, sizeof(float));

        _queue.enqueueWriteBuffer(this->_deviceVec, CL_TRUE, 0, this->_vec.size()*sizeof(float), this->_vec.data());
        _queue.enqueueWriteBuffer(o._deviceVec, CL_TRUE, 0, o._vec.size()*sizeof(float), o._vec.data());

        auto queue_plain = _queue();

        auto status = CLBlastSdot(
            this->_vec.size(),  // size
            device_c(),         // result
            0,                  // offset
            this->_deviceVec(), // x_buffer
            0,                  // x_offset
            1,                  // x_inc
            o._deviceVec(),     // y_buffer
            0,                  // y_offset
            1,                  // y_inc
            &queue_plain,       // queue
            &event              // event
        );

        if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }

        float result = 0.0f;
        _queue.enqueueReadBuffer(device_c, CL_TRUE, 0, sizeof(float), &result);

        return result;
    }
};

class MatrixCLBlast : public Matrix<float> {

protected:
    cl::Device _device;

public:
    MatrixCLBlast(Matrix<float> mat, cl::Device device = VectorOpenCL::defaultDevice) : Matrix<float>(mat), _device(device) { }

    void dot(const Matrix<float> &o, Matrix<float> &r) const {

        auto context = cl::Context(_device);
        auto queue = cl::CommandQueue(context, _device);
        auto event = cl_event{nullptr};

        auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, this->size()*sizeof(float));
        auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, o.size()*sizeof(float));
        auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, r.size()*sizeof(float));

        queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, this->size()*sizeof(float), this->data());
        queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, o.size()*sizeof(float), o.data());

        auto queue_plain = queue();

        auto status = CLBlastSgemm(
            CLBlastLayoutRowMajor, // layout
            CLBlastTransposeNo,    // a_transpose
            CLBlastTransposeNo,    // b_transpose
            this->rows(), // m
            o.cols(),     // n
            this->cols(), // k
            1.0f,         // alpha
            device_a(),   // a_buffer
            0,            // a_offset
            this->cols(), // a_ld
            device_b(),   // b_buffer
            0,            // b_offset
            o.cols(),     // b_ld
            0.0f,         // beta
            device_c(),   // c_buffer
            0,            // c_offset
            o.cols(),     // c_ld
            &queue_plain, // queue
            &event        // event
        );

        if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        queue.enqueueReadBuffer(device_c, CL_TRUE, 0, r.size()*sizeof(float), r.data());
    }
};

// CL_SUCCESS                                  0
// CL_DEVICE_NOT_FOUND                         -1
// CL_DEVICE_NOT_AVAILABLE                     -2
// CL_COMPILER_NOT_AVAILABLE                   -3
// CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
// CL_OUT_OF_RESOURCES                         -5
// CL_OUT_OF_HOST_MEMORY                       -6
// CL_PROFILING_INFO_NOT_AVAILABLE             -7
// CL_MEM_COPY_OVERLAP                         -8
// CL_IMAGE_FORMAT_MISMATCH                    -9
// CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
// CL_BUILD_PROGRAM_FAILURE                    -11
// CL_MAP_FAILURE                              -12
// CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
// CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
// CL_COMPILE_PROGRAM_FAILURE                  -15
// CL_LINKER_NOT_AVAILABLE                     -16
// CL_LINK_PROGRAM_FAILURE                     -17
// CL_DEVICE_PARTITION_FAILED                  -18
// CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19
// CL_INVALID_VALUE                            -30
// CL_INVALID_DEVICE_TYPE                      -31
// CL_INVALID_PLATFORM                         -32
// CL_INVALID_DEVICE                           -33
// CL_INVALID_CONTEXT                          -34
// CL_INVALID_QUEUE_PROPERTIES                 -35
// CL_INVALID_COMMAND_QUEUE                    -36
// CL_INVALID_HOST_PTR                         -37
// CL_INVALID_MEM_OBJECT                       -38
// CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
// CL_INVALID_IMAGE_SIZE                       -40
// CL_INVALID_SAMPLER                          -41
// CL_INVALID_BINARY                           -42
// CL_INVALID_BUILD_OPTIONS                    -43
// CL_INVALID_PROGRAM                          -44
// CL_INVALID_PROGRAM_EXECUTABLE               -45
// CL_INVALID_KERNEL_NAME                      -46
// CL_INVALID_KERNEL_DEFINITION                -47
// CL_INVALID_KERNEL                           -48
// CL_INVALID_ARG_INDEX                        -49
// CL_INVALID_ARG_VALUE                        -50
// CL_INVALID_ARG_SIZE                         -51
// CL_INVALID_KERNEL_ARGS                      -52
// CL_INVALID_WORK_DIMENSION                   -53
// CL_INVALID_WORK_GROUP_SIZE                  -54
// CL_INVALID_WORK_ITEM_SIZE                   -55
// CL_INVALID_GLOBAL_OFFSET                    -56
// CL_INVALID_EVENT_WAIT_LIST                  -57
// CL_INVALID_EVENT                            -58
// CL_INVALID_OPERATION                        -59
// CL_INVALID_GL_OBJECT                        -60
// CL_INVALID_BUFFER_SIZE                      -61
// CL_INVALID_MIP_LEVEL                        -62
// CL_INVALID_GLOBAL_WORK_SIZE                 -63
// CL_INVALID_PROPERTY                         -64
// CL_INVALID_IMAGE_DESCRIPTOR                 -65
// CL_INVALID_COMPILER_OPTIONS                 -66
// CL_INVALID_LINKER_OPTIONS                   -67
// CL_INVALID_DEVICE_PARTITION_COUNT           -68