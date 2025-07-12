
import time
import numpy as np

from intel_npu_acceleration_library.backend import MatMul


def timeit(call):
    start = time.perf_counter()
    res = call()
    t = round(time.perf_counter() - start, 4)

    return res, t


N, M = int(input('N: ')), int(input('M: '))

np.random.seed(42)
x = np.random.random((N, M))
y = x

F = 0.0001

x = x.astype(np.float16) * F
y = y.astype(np.float16) * F

print('Created: x', str(x.shape), 'y', str(y.shape), x.dtype)

inC = x.shape[1]
outC = y.shape[0]
batch = x.shape[0]


v, t = timeit(lambda: MatMul(inC, outC, batch).run(x, y).sum())
print('[Intel NPU]', 'Sum:', v, 'Time:', f'{t}s')
