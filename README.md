# Проект MatMul

## Описание

Проект представляет собой исследование производительности векторного и матричного умножения различными библиотеками. Для этого были использованы следующие библиотеки:
- C++ STL для обработки стандартных операций над векторами и матрицами;
- OpenBLAS для умножения, в частности, векторов и матриц, более быстрый путь чем STL;
- clBLAST для умножения с использованием библиотеки OpenCL.
- cuBLAS для умножения с использованием библиотеки CUDA.

## Используемые технологии
- C++14
- OpenCL
- OpenBLAS (https://github.com/OpenMathLib/OpenBLAS)
- clBLAST (https://github.com/CNugteren/CLBlast)
- cuBLAS (https://developer.nvidia.com/cuda-toolkit)

### Компиляция и запуск: 
```bash
g++.exe -g main.cpp -o example.exe -I. -I D:\3d\OpenCL-SDK-v2024.10.24-Win-x64\include -I D:\3d\CLBlast-1.6.3-windows-x64\include -I D:\3d\OpenBLAS-0.3.29_x64\include -L D:\3d\OpenCL-SDK-v2024.10.24-Win-x64\lib -L D:\3d\CLBlast-1.6.3-windows-x64\lib -L D:\3d\OpenBLAS-0.3.29_x64\lib -lopencl -lopenblas -lclblast -std=c++17
./example.exe
```

В данном проекте мы использовали библиотеки для векторного и матричного умножения, которые позволят сравнить производительность разных реализаций. В результате вы сможете увидеть, какая библиотека лучше всего работает для ваших задач.


