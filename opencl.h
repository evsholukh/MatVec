#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast_c.h>

#include "utils.h"
#include "matrix.h"


class OpenCL {

public:
    static cl::Context defaultContext() {
        auto device = defaultDevice();

        return cl::Context(device);
    }

    static std::string deviceName(cl::Device device) {
        return device.getInfo<CL_DEVICE_NAME>();
    }

    static std::string platformName(cl::Platform platform) {
        return platform.getInfo<CL_PLATFORM_NAME>(); 
    }

    static std::string deviceVersion(cl::Device device) {
        auto str = device.getInfo<CL_DEVICE_NAME>();
        Utils::rtrim(str);

        return str;
    }

    static std::string driverVersion(cl::Device device) {
        return device.getInfo<CL_DRIVER_VERSION>();
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
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    Vector<float> vec;
    cl::Buffer deviceVec;

    size_t blockSize, blocksCount, globalSize;

public:
    VectorOpenCL(
        Vector<float> vec,
        size_t blockSize = 256,
        size_t blocksCount = 128,
        cl::Device device = VectorOpenCL::defaultDevice,
        cl::Context context = VectorOpenCL::defaultContext,
        cl::CommandQueue queue = VectorOpenCL::defaultQueue
    ) : vec(vec),
        blockSize(blockSize),
        blocksCount(blocksCount),
        device(device),
        context(context),
        queue(queue) {

        globalSize = blockSize * blocksCount;
        deviceVec = cl::Buffer(context, CL_MEM_READ_ONLY, vec.size()*sizeof(float));
        queue.enqueueWriteBuffer(deviceVec, CL_TRUE, 0, vec.size()*sizeof(float), vec.data());

        program = cl::Program(context, kernel);
        program.build();
    }

    float dot(const VectorOpenCL &o) const {
        auto deviceResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, blocksCount*sizeof(float));
        auto addKernel = cl::Kernel(program, "float_dot_prod");

        addKernel.setArg(0, deviceVec);
        addKernel.setArg(1, o.deviceVec);
        addKernel.setArg(2, deviceResult);
        addKernel.setArg(3, blockSize * sizeof(float), nullptr);
        addKernel.setArg(4, static_cast<cl_uint>(vec.size())); 

        auto globalRange = cl::NDRange(globalSize);
        auto groupRange = cl::NDRange(blockSize);

        float *resData = new float[blocksCount];

        queue.enqueueNDRangeKernel(addKernel, cl::NullRange, globalRange, groupRange);
        queue.enqueueReadBuffer(deviceResult, CL_TRUE, 0, blocksCount * sizeof(float), resData);

        auto resVec = Vector<float>(resData, blocksCount);
        auto result = resVec.sum();

        delete[] resData;

        return result;
    }

    const std::string kernel = R"(
        __kernel void float_dot_prod(
            __global const float* a,
            __global const float* b,
            __global float* result,
            __local float* local_sum,
            const uint n)
        {
            uint gid = get_global_id(0);
            uint gsize = get_global_size(0);
            uint lid = get_local_id(0);
            uint l_size = get_local_size(0);

            float sum = 0.0f;
            for (uint i = gid; i < n; i += gsize) {
                sum += a[i] * b[i];
            }

            local_sum[lid] = sum;
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
        auto device_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float));
        auto queue_plain = queue();

        auto status = CLBlastSdot(
            vec.size(),         // size
            device_c(),         // result
            0,                  // offset
            deviceVec(),        // x_buffer
            0,                  // x_offset
            1,                  // x_inc
            o.deviceVec(),      // y_buffer
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
        queue.enqueueReadBuffer(device_c, CL_TRUE, 0, sizeof(float), &result);

        return result;
    }
};

class MatrixCLBlast {

protected:
    Matrix<float> mat;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Buffer deviceBuf;

public:
    MatrixCLBlast(
        Matrix<float> mat,
        cl::Device device = VectorOpenCL::defaultDevice,
        cl::Context context = VectorOpenCL::defaultContext,
        cl::CommandQueue queue = VectorOpenCL::defaultQueue
    ) : mat(mat),
        device(device),
        context(context),
        queue(queue) {
            deviceBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat.size()*sizeof(float), mat.data());
        }

    void dot(const MatrixCLBlast &o, MatrixCLBlast &r) const {
        auto event = cl_event{nullptr};
        auto queue_plain = queue();

        auto status = CLBlastSgemm(
            CLBlastLayoutRowMajor, // layout
            CLBlastTransposeNo,    // a_transpose
            CLBlastTransposeNo,    // b_transpose
            mat.rows(),            // m
            o.mat.cols(),          // n
            mat.cols(),            // k
            1.0f,                  // alpha
            deviceBuf(),           // a_buffer
            0,                     // a_offset
            mat.cols(),            // a_ld
            o.deviceBuf(),         // b_buffer
            0,                     // b_offset
            o.mat.cols(),          // b_ld
            0.0f,                  // beta
            r.deviceBuf(),         // c_buffer
            0,                     // c_offset
            o.mat.cols(),          // c_ld
            &queue_plain,          // queue
            &event                 // event
        );

        if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        queue.enqueueReadBuffer(r.deviceBuf, CL_TRUE, 0, r.mat.size()*sizeof(float), r.mat.data());
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