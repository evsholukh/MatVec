@echo off


nvcc cu_main.cu -o cu_example.exe ^
    -I"%CUDA_PATH_V12_4%\include" ^
    -L"%CUDA_PATH_V12_4%\lib\x64" ^
    -I%OPENCL_PATH%\include ^
    -L%OPENCL_PATH%\lib ^
    -lOpenCL ^
    -lcudart ^
    -allow-unsupported-compiler ^
    -std=c++17

cu_example.exe
