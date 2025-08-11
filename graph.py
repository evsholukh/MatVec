import subprocess
import json
import sys
import tqdm

total = []

block_sizes = [2**i for i in range(11)]
grid_sizes = [2**i for i in range(11)]
num_elemets = [10**8] #[10**i for i in range(7)]


pg = tqdm.tqdm(total=len(block_sizes)*len(grid_sizes)*len(num_elemets), file=sys.stderr)

for gs in grid_sizes:
    for bs in block_sizes:
        for sz in num_elemets:
            try:
                args = ["matmul_cuda.exe", "-s", str(sz), "-b", str(bs), "-g", str(gs)]
                print(args, file=sys.stderr)

                output = subprocess.check_output(args, timeout=120.0, start_new_session=True)
                data = json.loads(str(output, encoding="utf-8"))
                total += data

                print(output, file=sys.stderr)
                pg.update()
            except Exception as e:
                print(str(e), file=sys.stderr)
                continue

json.dump(total, sys.stdout, indent=2)
