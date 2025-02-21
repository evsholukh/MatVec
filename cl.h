#include <iostream>
#include <random>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>


class OpenCLHelper {

private:
    cl::Platform platform;
    cl::Device device;
    cl::Context *context;
    cl::CommandQueue *queue;
    cl::Program *program;

    static const std::string source;
    std::string decodeError(cl_int error);

public:
    OpenCLHelper();
    ~OpenCLHelper();

    void vector_sum(std::vector<float> &x, std::vector<float> &y);
    float vector_scalar_product(std::vector<float> &x, std::vector<float> &y);

    void close();
    void print_info();
};

OpenCLHelper::OpenCLHelper() {
    std::vector<cl::Platform> platforms;
    cl_int err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Getting platforms error: " + this->decodeError(err));
    }
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }
    this->platform = platforms.back();

    std::vector<cl::Device> devices;
    err = this->platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Getting devices error: " + this->decodeError(err));
    }
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found.");
    }
    this->device = devices.back();
    this->context = new cl::Context(this->device);
    this->queue = new cl::CommandQueue(*this->context, this->device);

    this->program = new cl::Program(*this->context, this->source);
    err = this->program->build();
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Compilation error: " + this->decodeError(err));
    }
}

OpenCLHelper::~ OpenCLHelper() {}

inline void OpenCLHelper::vector_sum(std::vector<float> &x, std::vector<float> &y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Invalid vector sizes " + std::to_string(x.size()) + " " + std::to_string(y.size()));
    }
    size_t bytes_count = sizeof(float) * x.size();
    cl_int err;

    // Создание буферов на устройстве
    cl::Buffer buffer_x(*this->context, CL_MEM_READ_WRITE, bytes_count);
    cl::Buffer buffer_y(*this->context, CL_MEM_READ_ONLY, bytes_count);

    // Копирование массивов на устройство
    err = this->queue->enqueueWriteBuffer(buffer_x, CL_TRUE, 0, bytes_count, x.data());
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Copying buf to device error: "+ this->decodeError(err));
    }
    err = this->queue->enqueueWriteBuffer(buffer_y, CL_TRUE, 0, bytes_count, y.data());
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Copying buf to device error: "+ this->decodeError(err));
    }

    // Задание аргументов ядра
    cl::Kernel kernel(*this->program, "vector_add");
    err = kernel.setArg(0, buffer_x);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Setting arg error: " + this->decodeError(err));
    }
    err = kernel.setArg(1, buffer_y);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Setting arg error: " + this->decodeError(err));
    }

    // Добавление в очередь запуск ядра
    cl::NDRange global_range(x.size());
    err = this->queue->enqueueNDRangeKernel(kernel, cl::NullRange, global_range, cl::NullRange);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Enquening kernel error: " + this->decodeError(err));
    }

    // Копирование результатов обратно в память хоста
    err = this->queue->enqueueReadBuffer(buffer_x, CL_TRUE, 0, bytes_count, x.data());
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Copying array from device error: " + this->decodeError(err));
    }
}

inline void OpenCLHelper::close() {
    // cl_int err = this->queue->finish();
    // if (err != CL_SUCCESS) {
    //     throw std::runtime_error("Finishing queue error: " + this->decodeError(err));
    // }
}

inline void OpenCLHelper::print_info() {
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
    std::cout << "GPU memory available: " << mem_size / (1024 * 1024) << " MB" << std::endl;
}

const std::string OpenCLHelper::source = R"(
    __kernel void vector_add(__global float* a, __global const float* b) {
        int i = get_global_id(0);
        a[i] = a[i] + b[i];
    }
)";


std::string OpenCLHelper::decodeError(cl_int err) {
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
