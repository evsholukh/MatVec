import subprocess
import json
import sys

# Vector
argv_vars = [
    ["vector.exe", "--correct"],
    ["vector.exe", "--cpu"],
    ["vector.exe", "--openmp"],
    ["vector.exe", "--openblas"],
    ["vector.exe", "--opencl"],
    ["vector.exe", "--clblast"],
    ["vector_cuda.exe", "--cuda"],
    ["vector_cuda.exe", "--cublas"],
]

# Matrix
argm_vars= [    
    ["matrix.exe", "--cpu"],
    ["matrix.exe", "--openblas"],
    ["matrix.exe", "--clblast"],
    ["matrix_cuda.exe", "--cublas"],
]

N = 1
total = []
dtypes = ["--float", "--double"]
sizes = [10**i for i in range(2, 9)]

i = 0
j = N * len(argv_vars) * len(dtypes) * len(sizes)

for _ in range(N):
    for av in argv_vars:
        for dtype in dtypes:
            for size in sizes:
                try:
                    i += 1
                    args = [*av, "--seed", str(42), dtype, "--size", str(size)]

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
