
import numpy as np
import time


def timeit(call):
    start = time.perf_counter()
    res = call()
    t = round(time.perf_counter() - start, 4)

    return res, t


def single_cpu_vector_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.zeros((x.shape[0], y.shape[1]))

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            s = 0
            for k in range(x.shape[1]):
                s += x[i, k] * y[k, j]
            z[i, j] = s
    return z


N, M = int(input('N: ')), int(input('M: '))
x = np.zeros((N, M)) + 0.0001
y = x
print('Created: x', str(x.shape), 'y', str(y.shape))

v, t = timeit(lambda: single_cpu_vector_dot(x, y.T).sum())
print('[CPU]', 'Sum:', v, 'Time:', f'{t}s')

v, t = timeit(lambda: np.dot(x, y.T).sum())
print('[CPU Multi]', 'Sum:', v, 'Time:', f'{t}s')

