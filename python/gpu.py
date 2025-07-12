
import time
import torch
import numpy as np


def timeit(call):
    start = time.perf_counter()
    res = call()
    t = round(time.perf_counter() - start, 4)

    return res, t

assert torch.xpu.is_available()

N, M = int(input('N: ')), int(input('M: '))

np.random.seed(42)
x = np.random.random((N, M))
y = x

print('Created: x', str(x.shape), 'y', str(y.shape), x.dtype)

device = torch.device('xpu')

F = 0.0001

tx = torch.tensor(x).to(device) * F
ty = torch.tensor(y).to(device) * F

v, t = timeit(lambda: tx.matmul(ty.T).cpu().numpy().sum())
print('[Intel GPU]', 'Sum:', v, 'Time:', f'{t}s')

