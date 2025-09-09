import subprocess
import json
import sys
import tqdm

T = 100_000_000
N = [i for i in range(T, T*10, T)] # 8*(10**8)
progs = ["vectors.exe", "vectors_cuda.exe"]

total = []
for exe in progs:
    for n in tqdm.tqdm(N):
        try:
            args = [exe, "-n", str(n), "-b", str(1024), "-g", str(1024*1024)]
            # print(*args)
            output = subprocess.check_output(args, timeout=1200.0)
            data = json.loads(str(output, encoding="utf-8"))
            # json.dump(data, sys.stderr, indent=2)
            total += data
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(str(e))
            break

json.dump(total, open("metrics.json", "w"), indent=2)