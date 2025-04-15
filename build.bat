@echo off

set PATH=%PATH%;OpenCL-SDK-v2024.10.24-Win-x64\bin;C:\msys64\mingw64\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

g++ cl_main.cpp -o cl_example.exe ^
    -IOpenCL-SDK-v2024.10.24-Win-x64\include ^
    -LOpenCL-SDK-v2024.10.24-Win-x64\lib ^
    -lOpenCL

if %errorlevel% neq 0 exit /b %errorlevel%

nvcc cu_main.cu -o cu_example.exe ^
    -I"%CUDA_PATH_V12_4%\include" ^
    -L"%CUDA_PATH_V12_4%\lib\x64" ^
    -lcudart ^
    -allow-unsupported-compiler ^
    -std=c++17

if %errorlevel% neq 0 exit /b %errorlevel%

cl_example.exe

if %errorlevel% neq 0 exit /b %errorlevel%

cu_example.exe
