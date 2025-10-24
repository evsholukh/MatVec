import subprocess
import json

dots_cpu = [
    ["dot.exe", "--openmp"],
    ["dot.exe", "--openblas"],
]

dots_gpu = [
    ["dot.exe", "--opencl"],
    ["dot.exe", "--clblast"],
    ["dot_cuda.exe", "--cuda"],
    ["dot_cuda.exe", "--cublas"],
]

gemms_cpu = [
    ["gemm.exe", "--openmp"],
    ["gemm.exe", "--openblas"],
]

gemms_gpu = [
    ["gemm.exe", "--opencl"],
    ["gemm.exe", "--clblast"],
    ["gemm_cuda.exe", "--cuda"],
    ["gemm_cuda.exe", "--cublas"],
]

R = 1
dtypes = ["--float", "--double"]
mnk = [2**i for i in range(10, 15)]

total = []

i = 0
N = (len(dots_gpu) + len(gemms_gpu)) * len(dtypes) * len(mnk)

for _ in range(R):
    for dot in dots_gpu:
        for dtype in dtypes:
            for n in mnk:
                try:
                    i += 1
                    args = [*dot, dtype, "-n", str(n)]

                    print(f"[{i}/{N}]", *args)
                    output = subprocess.check_output(args, timeout=2400.0)
                    data = json.loads(str(output, encoding="utf-8"))
                    total.append(data)
                except KeyboardInterrupt:
                    print("Ctrl+C")
                    break
                except Exception as e:
                    print(str(e))
                    continue

for _ in range(R):
    for gemm in gemms_gpu:
        for dtype in dtypes:
            for n in mnk:
                try:
                    i += 1
                    args = [*gemm, dtype, "-m", str(n), "-n", str(n), "-k", str(n)]

                    print(f"[{i}/{N}]", *args)
                    output = subprocess.check_output(args, timeout=2400.0)
                    data = json.loads(str(output, encoding="utf-8"))
                    total.append(data)
                except KeyboardInterrupt:
                    print("Ctrl+C")
                    break
                except Exception as e:
                    print(str(e))
                    continue

with open("bench.json", "w") as f:
    json.dump(total, f, indent=2)
