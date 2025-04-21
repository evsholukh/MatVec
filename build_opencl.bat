@echo off


g++ cl_main.cpp -o cl_example.exe ^
    -I%OPENCL_PATH%\include ^
    -L%OPENCL_PATH%\lib ^
    -lOpenCL

if %errorlevel% neq 0 exit /b %errorlevel%

cl_example.exe
