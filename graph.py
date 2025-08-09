import subprocess
import json
import os
import sys

total = []

step = 10_000
blocks_count = [j for j in range(step, 500000, step)]
block_sizes = [2**j for j in range(2, 11)]
num_elemets = [u*v for u, v in zip(block_sizes, blocks_count)]


for i in blocks_count:
    for j in block_sizes:
        try:
            args = ["matmul.exe", "-n", str(i), "-b", str(j)]
            print(args, file=sys.stderr)

            output = subprocess.check_output(args, timeout=120.0, start_new_session=True)
            data = json.loads(str(output, encoding="utf-8"))
            total += data
        except Exception as e:
            print(str(e), file=sys.stderr)
            continue

json.dump(total, sys.stdout, indent=2)
