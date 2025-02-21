#include <iostream>
#include <random>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>

using namespace std;


#define CHECK_OPENCL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        cerr << msg << " Error code: " << err << endl; \
        return 1; \
    }


const char* kernel_source = R"(
    __kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
)";


int main(int argc, char **argv) {
    const int ARRAY_SIZE = 100000000;
    const int seed = 42;

    cout << "Vector size " << fixed << ARRAY_SIZE << endl;
    cout << "Generating random vectors.." << endl;

    mt19937 generator(seed);
    uniform_real_distribution<float> dist(-1, 1);
    vector<float> a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE);
    float res = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = dist(generator);
        b[i] = dist(generator);
    }
    cout << "Running C++.." << endl;
    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "Elapsed time: " << fixed << elapsed.count() << " seconds" << endl;

    vector<cl::Platform> platforms;
    cl_int err = cl::Platform::get(&platforms);
    CHECK_OPENCL_ERROR(err, "Getting platform error");
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found!" << endl;
        return 1;
    }

    cl::Platform platform = platforms.back();
    string platform_name;
    err = platform.getInfo(CL_PLATFORM_NAME, &platform_name);
    CHECK_OPENCL_ERROR(err, "Getting platform info error");
    cout << "Platform: " << platform_name << endl;

    vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    CHECK_OPENCL_ERROR(err, "Getting devices error");
    if (devices.empty()) {
        cerr << "No GPU devices found!" << endl;
        return 1;
    }

    cl::Device device = devices.back();
    string device_name;
    err = device.getInfo(CL_DEVICE_NAME, &device_name);
    CHECK_OPENCL_ERROR(err, "Getting device info error");
    cout << "Device: " << device_name << endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a.data(), &err);
    CHECK_OPENCL_ERROR(err, "Creating buffer_a error");

    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b.data(), &err);
    CHECK_OPENCL_ERROR(err, "Creating buffer_b error");

    cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * ARRAY_SIZE, nullptr, &err);
    CHECK_OPENCL_ERROR(err, "Creating buffer_c error");

    cl::Program program(context, kernel_source);
    err = program.build();
    CHECK_OPENCL_ERROR(err, "Building program error");

    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, buffer_a);
    kernel.setArg(1, buffer_b);
    kernel.setArg(2, buffer_c);

    cout << "Running OpenCL kernel.." << endl;
    cl::NDRange global_size(ARRAY_SIZE);

    start_time = chrono::high_resolution_clock::now();
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
    CHECK_OPENCL_ERROR(err, "Enquening kernel error");

    vector<float> d(ARRAY_SIZE);
    err = queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, d.data());
    CHECK_OPENCL_ERROR(err, "Enquening reading error");

    end_time = chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    cout << "Elapsed time: " << fixed << elapsed.count() << " seconds" << endl;

    err = queue.finish();
    CHECK_OPENCL_ERROR(err, "Finishing queue error");

    float mse = 0.0f;
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        mse += pow(d[i] - c[i], 2);
    }
    cout << "MSE: "<< fixed << mse << endl;
    return 0;
}