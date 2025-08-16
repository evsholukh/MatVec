import subprocess
import json


total = []
progs = ["matmul.exe", "matmul_cuda.exe"]

for exe in progs:
    try:
        args = [exe, "-s", str(10**8), "-b", str(1024), "-g", str(1000000)]
        print(*args)
        output = subprocess.check_output(args, timeout=120.0)
        data = json.loads(str(output, encoding="utf-8"))
        total += data
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(str(e))
        continue

json.dump(total, open("metrics.json", "w"), indent=2)