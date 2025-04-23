@echo off


nvcc main.cu -o cu_example.exe ^
    -D BUILD_CUDA ^
    -I"%CUDA_PATH_V12_4%\include" ^
    -L"%CUDA_PATH_V12_4%\lib\x64" ^
    -I%OPENCL_PATH%\include ^
    -L%OPENCL_PATH%\lib ^
    -lOpenCL ^
    -lcudart ^
    -lcublas ^
    -allow-unsupported-compiler ^
    -std=c++17

if %errorlevel% neq 0 exit /b %errorlevel%

cu_example.exe
