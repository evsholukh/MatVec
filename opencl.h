#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast_c.h>
#include <sstream> 

#include "utils.h"
#include "matrix.h"


class OpenCL {

public:
    static cl::Context defaultContext() {
        auto device = defaultDevice();

        return cl::Context(device);
    }

    static std::string getDeviceName(cl::Device device) {
        return device.getInfo<CL_DEVICE_NAME>();
    }

    static std::string getPlatformName(cl::Platform platform) {
        return platform.getInfo<CL_PLATFORM_NAME>(); 
    }

    static std::string getDeviceVersion(cl::Device device) {
        auto str = device.getInfo<CL_DEVICE_VERSION>();
        Utils::rtrim(str);

        return str;
    }

    static std::string getDriverVersion(cl::Device device) {
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

    static std::string getCompilerVersion() {
        return __VERSION__;
    }
};

template <typename T = void>
class VectorOpenCL : public Vector<T> {

public:
    static cl::Device defaultDevice;
    static cl::Context defaultContext;
    static cl::CommandQueue defaultQueue;

protected:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Buffer deviceVec;

    size_t blockSize, blocksCount, globalSize;

    static const std::string source;

public:
    VectorOpenCL(
        Vector<T> vec,
        size_t blockSize = 256,
        size_t blocksCount = 128,
        cl::Device device = VectorOpenCL::defaultDevice,
        cl::Context context = VectorOpenCL::defaultContext,
        cl::CommandQueue queue = VectorOpenCL::defaultQueue
    ) : Vector<T>(vec),
        blockSize(blockSize),
        blocksCount(blocksCount),
        device(device),
        context(context),
        queue(queue) {

        globalSize = blockSize * blocksCount;
        deviceVec = cl::Buffer(context, CL_MEM_READ_ONLY, vec.size()*sizeof(T));
        queue.enqueueWriteBuffer(deviceVec, CL_TRUE, 0, vec.size()*sizeof(T), vec.data());

        program = cl::Program(context, source);
        program.build();
    }

    T dot (const Vector<T> &o) const {
        // [!]
        if (auto* ocl = static_cast<const VectorOpenCL<T>*>(&o)) {
            return this->dot(*ocl);
        }
        return Vector<T>::dot(o);
    }

    virtual T dot(const VectorOpenCL<T> &o) const;
};

template <typename T>
const std::string VectorOpenCL<T>::source = R"(
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

    __kernel void double_dot_prod(
        __global const double* a,
        __global const double* b,
        __global double* result,
        __local double* local_sum,
        const uint n)
    {
        uint gid = get_global_id(0);
        uint gsize = get_global_size(0);
        uint lid = get_local_id(0);
        uint l_size = get_local_size(0);

        double sum = 0.0f;
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

template <>
float VectorOpenCL<float>::dot(const VectorOpenCL<float> &o) const {
    auto deviceResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, blocksCount*sizeof(float));
    auto addKernel = cl::Kernel(program, "float_dot_prod");

    addKernel.setArg(0, deviceVec);
    addKernel.setArg(1, o.deviceVec);
    addKernel.setArg(2, deviceResult);
    addKernel.setArg(3, blockSize * sizeof(float), nullptr);
    addKernel.setArg(4, static_cast<cl_uint>(this->size())); 

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

template <>
double VectorOpenCL<double>::dot(const VectorOpenCL<double> &o) const {
    auto deviceResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, blocksCount*sizeof(double));
    auto addKernel = cl::Kernel(program, "double_dot_prod");

    addKernel.setArg(0, deviceVec);
    addKernel.setArg(1, o.deviceVec);
    addKernel.setArg(2, deviceResult);
    addKernel.setArg(3, blockSize * sizeof(double), nullptr);
    addKernel.setArg(4, static_cast<cl_uint>(this->size())); 

    auto globalRange = cl::NDRange(globalSize);
    auto groupRange = cl::NDRange(blockSize);

    double *resData = new double[blocksCount];

    queue.enqueueNDRangeKernel(addKernel, cl::NullRange, globalRange, groupRange);
    queue.enqueueReadBuffer(deviceResult, CL_TRUE, 0, blocksCount * sizeof(double), resData);

    auto resVec = Vector<double>(resData, blocksCount);
    auto result = resVec.sum();

    delete[] resData;

    return result;
}

template <typename T>
cl::Device VectorOpenCL<T>::defaultDevice = OpenCL::defaultDevice();

template <typename T>
cl::Context VectorOpenCL<T>::defaultContext = OpenCL::defaultContext();

template <typename T>
cl::CommandQueue VectorOpenCL<T>::defaultQueue = cl::CommandQueue(VectorOpenCL::defaultContext, VectorOpenCL::defaultDevice);


template <typename T = void>
class VectorCLBlast : public VectorOpenCL<T> {

public:
    VectorCLBlast(Vector<T> vec) : VectorOpenCL<T>(vec) {}

    T dot (const Vector<T> &o) const {
        // [!]
        if (auto* ocl = static_cast<const VectorCLBlast<T>*>(&o)) {
            return this->dot(*ocl);
        }
        return Vector<T>::dot(o);
    }

    virtual T dot(const VectorCLBlast<T> &o) const;

    static std::string getCLBlastVersion() {
        std::stringstream ss;

        ss << CLBLAST_VERSION_MAJOR << "." << CLBLAST_VERSION_MINOR << "." << CLBLAST_VERSION_PATCH;
        return ss.str();
    }
};

template <>
float VectorCLBlast<float>::dot(const VectorCLBlast<float> &o) const {
    auto event = cl_event{nullptr};
    auto device_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float));
    auto queue_plain = queue();

    auto status = CLBlastSdot(
        this->size(),         // size
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

template <>
double VectorCLBlast<double>::dot(const VectorCLBlast<double> &o) const {
    auto event = cl_event{nullptr};
    auto device_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(double));
    auto queue_plain = queue();

    auto status = CLBlastDdot(
        this->size(),         // size
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

    double result = 0.0f;
    queue.enqueueReadBuffer(device_c, CL_TRUE, 0, sizeof(double), &result);

    return result;
}


template <typename T>
class MatrixCLBlast : public Matrix<T> {

protected:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Buffer deviceBuf;

public:
    MatrixCLBlast(
        Matrix<T> mat,
        cl::Device device = VectorOpenCL<T>::defaultDevice,
        cl::Context context = VectorOpenCL<T>::defaultContext,
        cl::CommandQueue queue = VectorOpenCL<T>::defaultQueue
    ) : Matrix<T>(mat),
        device(device),
        context(context),
        queue(queue) {
            deviceBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat.size()*sizeof(T), mat.data());
        }

    void gemm(const Matrix<T> &o, Matrix<T> &r) const override {
        // [!]
        if (auto* ocl = static_cast<const MatrixCLBlast<T>*>(&o)) {
            if (auto* rcl = static_cast<MatrixCLBlast<T>*>(&r)) {
                this->gemm(*ocl, *rcl);
                return;
            }
        }
        Matrix<T>::gemm(o, r);
    }

    virtual void gemm(const MatrixCLBlast<T> &o, MatrixCLBlast<T> &r) const;
};

template <>
void MatrixCLBlast<float>::gemm(const MatrixCLBlast<float> &o, MatrixCLBlast<float> &r) const {
    auto event = cl_event{nullptr};
    auto queue_plain = queue();

    auto status = CLBlastSgemm(
        CLBlastLayoutRowMajor, // layout
        CLBlastTransposeNo,    // a_transpose
        CLBlastTransposeNo,    // b_transpose
        this->rows(),          // m
        o.cols(),              // n
        this->cols(),          // k
        1.0f,                  // alpha
        deviceBuf(),           // a_buffer
        0,                     // a_offset
        this->cols(),          // a_ld
        o.deviceBuf(),         // b_buffer
        0,                     // b_offset
        o.cols(),              // b_ld
        0.0f,                  // beta
        r.deviceBuf(),         // c_buffer
        0,                     // c_offset
        o.cols(),              // c_ld
        &queue_plain,          // queue
        &event                 // event
    );

    if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    queue.enqueueReadBuffer(r.deviceBuf, CL_TRUE, 0, r.size()*sizeof(float), r.data());
}

template <>
void MatrixCLBlast<double>::gemm(const MatrixCLBlast<double> &o, MatrixCLBlast<double> &r) const {
    auto event = cl_event{nullptr};
    auto queue_plain = queue();

    auto status = CLBlastDgemm(
        CLBlastLayoutRowMajor, // layout
        CLBlastTransposeNo,    // a_transpose
        CLBlastTransposeNo,    // b_transpose
        this->rows(),          // m
        o.cols(),              // n
        this->cols(),          // k
        1.0f,                  // alpha
        deviceBuf(),           // a_buffer
        0,                     // a_offset
        this->cols(),          // a_ld
        o.deviceBuf(),         // b_buffer
        0,                     // b_offset
        o.cols(),              // b_ld
        0.0f,                  // beta
        r.deviceBuf(),         // c_buffer
        0,                     // c_offset
        o.cols(),              // c_ld
        &queue_plain,          // queue
        &event                 // event
    );

    if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    queue.enqueueReadBuffer(r.deviceBuf, CL_TRUE, 0, r.size()*sizeof(double), r.data());
}

template <typename T>
class MatrixOpenCL : public MatrixCLBlast<T> {

private:
    cl::Program program;

public:
    MatrixOpenCL(
        Matrix<T> mat,
        cl::Device device = VectorOpenCL<>::defaultDevice,
        cl::Context context = VectorOpenCL<>::defaultContext,
        cl::CommandQueue queue = VectorOpenCL<>::defaultQueue
    ) : MatrixCLBlast<T>(mat, device, context, queue) {

        program = cl::Program(context, source);
        program.build();
    }

    void gemm(const Matrix<T> &o, Matrix<T> &r) const override {
        // [!]
        if (auto* ocl = static_cast<const MatrixOpenCL<T>*>(&o)) {
            if (auto* rcl = static_cast<MatrixOpenCL<T>*>(&r)) {
                this->gemm(*ocl, *rcl);
                return;
            }
        }
        Matrix<T>::gemm(o, r);
    }

    virtual void gemm(const MatrixOpenCL<T> &o, MatrixOpenCL<T> &r) const;

protected:
    static const std::string source;
};

template <typename T>
void MatrixOpenCL<T>::gemm(const MatrixOpenCL<T> &o, MatrixOpenCL<T> &r) const {
    
    const int M = this->rows();
    const int N = o.cols();
    const int K = this->cols();

    auto kernel = cl::Kernel(program, "GEMM");

    kernel.setArg(0, M);
    kernel.setArg(1, N);
    kernel.setArg(2, K);
    kernel.setArg(3, this->deviceBuf);
    kernel.setArg(4, o.deviceBuf);
    kernel.setArg(5, r.deviceBuf);

    const int B = 32;
    auto globalSizes = cl::NDRange(
        ((M + (B - 1)) / B) * B,
        ((N + (B - 1)) / B) * B
    );
    auto localSizes = cl::NDRange(B, B);

    cl::Event event;
    auto status = this->queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, localSizes, nullptr, &event);

    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue SGEMM kernel");
    }
    event.wait();

    status = this->queue.enqueueReadBuffer(r.deviceBuf, CL_TRUE, 0, r.size() * sizeof(T), r.data(), nullptr, &event);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue SGEMM kernel");
    }
    event.wait();
}

template<>
const std::string MatrixOpenCL<float>::source = R"(
    __kernel void GEMM(
        const int M,
        const int N,
        const int K,
        const __global float *A,
        const __global float *B,
        __global float *C)
    {
        const int global_row_index = get_global_id(0);
        const int global_col_index = get_global_id(1);

        if (global_row_index >= M || global_col_index >= N) {
            return;
        }
        // printf("global_row_index=%d, global_col_index=%d\n", global_row_index, global_col_index);

        float c = 0.f;
        for (int k = 0; k < K; k++) {
            c += A[global_row_index * K + k] * B[k * N + global_col_index];
        }
        C[global_row_index * N + global_col_index] = c;
    }
)";

template<>
const std::string MatrixOpenCL<double>::source = R"(
    __kernel void GEMM(
        const int M,
        const int N,
        const int K,
        const __global double *A,
        const __global double *B,
        __global double *C) 
    {
        const int global_row_index = get_global_id(0);
        const int global_col_index = get_global_id(1);

        if (global_row_index >= M || global_col_index >= N) {
            return;
        }
        // printf("global_row_index=%d, global_col_index=%d\n", global_row_index, global_col_index);

        double c = 0.0;
        for (int k = 0; k < K; k++) {
            c += A[global_row_index * K + k] * B[k * N + global_col_index];
        }
        C[global_row_index * N + global_col_index] = c;
    }
)";

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