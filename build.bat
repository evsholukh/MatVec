

set PATH=%PATH%;OpenCL-SDK-v2024.10.24-Win-x64\bin

g++ main.cpp -o cl_example.exe -IOpenCL-SDK-v2024.10.24-Win-x64\include -LOpenCL-SDK-v2024.10.24-Win-x64\lib -lOpenCL

cl_example.exe