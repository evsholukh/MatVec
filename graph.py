import subprocess
import json
import sys

arg_vars = [
    # CPU
    ["vector.exe", "--cpu"],
    ["vector_debug.exe", "--cpu"],
    ["vector.exe", "--openmp"],
    ["vector_debug.exe", "--openmp"],
    ["vector.exe", "--openblas"],
    ["vector_debug.exe", "--openblas"],

    # GPU
    ["vector.exe", "--opencl"],

    # GPU
    ["vector_debug.exe", "--opencl"],
    ["vector.exe", "--clblast"],
    ["vector_debug.exe", "--clblast"],
    ["vector_cuda.exe", "--cuda"],
    ["vector_cuda_debug.exe", "--cuda"],
    ["vector_cuda.exe", "--cublas"],
    ["vector_cuda_debug.exe", "--cublas"],

    # Matrix
    ["matrix.exe", "--cpu"],

    # Matrix
    ["matrix_debug.exe", "--cpu"],
    ["matrix.exe", "--openblas"],
    ["matrix_debug.exe", "--openblas"],
    ["matrix.exe", "--clblast"],
    ["matrix_debug.exe", "--clblast"],
    ["matrix_cuda.exe", "--cublas"],
    ["matrix_cuda_debug.exe", "--cublas"],
]

N = 1
total = []

i = 0
j = N * len(arg_vars)
for _ in range(N):
    for args in arg_vars:
        try:
            i += 1
            print(f"[{i}/{j}]", *args)
            output = subprocess.check_output(args, timeout=1200.0)
            data = json.loads(str(output, encoding="utf-8"))
            total.append(data)
        except KeyboardInterrupt:
            print("Ctrl+C")
            sys.exit(-1)
        except Exception as e:
            print(str(e))

json.dump(total, open("metrics.json", "w"), indent=2)
