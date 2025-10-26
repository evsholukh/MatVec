import subprocess
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

cpu_flags = ["--openmp", "--openblas"]
gpu_flags = ["--opencl", "--clblast", "--cuda", "--cublas"]
exec_dot = ["dot.exe", "dot_cuda.exe"]
exec_gemm = ["gemm.exe", "gemm_cuda.exe"]
dtypes = ["--float", "--double"]
seed_flags = ["--seed", "42"]

cpu_dot_n = [str(10**3*i) for i in range(1, 9)]

cpu_gemm_n = [str(2**i) for i in range(4, 7)]
gpu_gemm_n = [str(2**i) for i in range(4, 7)]

dot_cpu_cmds = []
for size in cpu_dot_n:
    for flag in cpu_flags:
        for exe in exec_dot:
            for dtype in dtypes:
                cmd = [exe, "-n", size, flag, dtype, *seed_flags]
                dot_cpu_cmds.append(cmd)

gemm_cpu_cmds = []
for size in cpu_gemm_n:
    for flag in cpu_flags:
        for exe in exec_gemm:
            for dtype in dtypes:
                cmd = [exe, "-n", size, "-m", size, "-k", size, flag, dtype, *seed_flags]
                gemm_cpu_cmds.append(cmd)

dot_gpu_cmds = []
for size in cpu_dot_n:
    for flag in gpu_flags:
        for exe in exec_dot:
            for dtype in dtypes:
                cmd = [exe, "-n", size, flag, dtype, *seed_flags]
                dot_gpu_cmds.append(cmd)

gemm_gpu_cmds = []
for size in gpu_gemm_n:
    for flag in gpu_flags:
        for exe in exec_gemm:
            for dtype in dtypes:
                cmd = [exe, "-n", size, "-m", size, "-k", size, flag, dtype, *seed_flags]
                gemm_gpu_cmds.append(cmd)


def run_json(args):
    output = subprocess.check_output(args, timeout=2400.0)
    data = json.loads(str(output, encoding="utf-8"))

    return data

def run_json_bulk(*args_arr):
    total = []
    N = 10
    for k, v in enumerate(args_arr):
        try:
            logging.info(f'[{k+1}/{len(args_arr)}] {" ".join(v)}', )
            for _ in range(N):
                res = run_json(v)
                total.append(res)
        except KeyboardInterrupt:
            logging.warning("Ctrl+C")
            break
        except Exception as e:
            logging.error(str(e), exc_info=False)
    return total

total = dot_cpu_cmds + dot_gpu_cmds + gemm_cpu_cmds + gemm_gpu_cmds

res = run_json_bulk(*total)

with open("bench.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=4)
