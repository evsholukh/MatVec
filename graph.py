import subprocess
import json
import sys
import tqdm

total = []

block_sizes = [2**i for i in range(4, 11)]
grid_sizes = [10**i for i in range(4, 9)]
num_elemets = [10**8]
progs = ["matmul.exe", "matmul_cuda.exe"]

pg = tqdm.tqdm(total=len(block_sizes)*len(grid_sizes)*len(num_elemets)*len(progs))

for gs in grid_sizes:
    for bs in block_sizes:
        for sz in num_elemets:
            for exe in progs:
                try:
                    args = [exe, "-s", str(sz), "-b", str(bs), "-g", str(gs)]
                    print(*args)
                    output = subprocess.check_output(args, timeout=120.0)
                    data = json.loads(str(output, encoding="utf-8"))
                    total += data

                    pg.update()
                    print()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(str(e))
                    continue

json.dump(total, open("metrics.json", "w"), indent=2)