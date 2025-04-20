@echo off


g++ main.cpp -o cl_example.exe ^
    -I%OPENCL_PATH%\include ^
    -L%OPENCL_PATH%\lib ^
    -lOpenCL

cl_example.exe
