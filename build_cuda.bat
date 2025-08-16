@echo off

nvcc main.cu -o matmul_cuda.exe ^
    -I"%CUDA_PATH_V12_4%\include" ^
    -L"%CUDA_PATH_V12_4%\lib\x64" ^
    -lcudart ^
    -lcublas ^
    -allow-unsupported-compiler ^
    -std=c++17
