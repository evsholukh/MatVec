
## MatVec

Этот проект реализует **бенчмарки операций над векторами и матрицами** с использованием различных технологий вычислений:

* Чистый C++
* OpenMP
* OpenBLAS
* OpenCL
* clBLASt
* CUDA
* cuBLAS

Результаты измерений сохраняются в формате **JSON** для удобного анализа.
---

## 🚀 Использование

### Векторы

Пример запуска:

```bash
./vector -n 1000000 --cpu
```
### Матрицы

Пример запуска:

```bash
./matrix -r 1000 -c 1000 --openblas
```

## 📊 Пример вывода

Результаты выводятся в формате JSON:

```json
{
    "block_size": 1024,
    "grid_size": 32768,
    "hardware": {
        "cpu": "Intel(R) Core(TM) Ultra 7 155H",
        "gpu": "Intel(R) Arc(TM) Graphics"
    },
    "max": 1.0,
    "min": -1.0,
    "result": 33329568.560428713,
    "runtime": [
        {
            "name": "C++",
            "optimization": "-O3",
            "standard": "C++17",
            "version": "13.3.0"
        },
        {
            "name": "OpenMP",
            "version": "4.5"
        },
        {
            "name": "OpenBLAS",
            "version": "OpenBLAS 0.3.29"
        },
        {
            "driver": "31.0.101.5382",
            "name": "OpenCL",
            "platform": "Intel(R) OpenCL Graphics",
            "version": "OpenCL 3.0 NEO"
        },
        {
            "name": "CLBlast",
            "version": "1.6.3"
        }
    ],
    "seed": 42,
    "size": 100000000,
    "tests": [
        {
            "duration": 87.08979797363281,
            "result": 33329568.56062876,
            "runtime": "C++"
        },
        {
            "duration": 33.437801361083984,
            "result": 33329568.56047061,
            "runtime": "OpenMP"
        },
        {
            "duration": 25.69059944152832,
            "result": 33329568.560438026,
            "runtime": "OpenBLAS"
        },
        {
            "duration": 39.6343994140625,
            "result": 33329568.560428854,
            "runtime": "OpenCL"
        },
        {
            "duration": 35.8026008605957,
            "result": 33329568.560428865,
            "runtime": "CLBlast"
        }
    ],
    "type": "double"
}
```
