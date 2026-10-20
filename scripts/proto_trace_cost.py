"""Biaya update jejak RTRL: 3 x d^2 tulis + baca, per lapisan (dominasi fwd?)."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from numba import njit
P = time.perf_counter

@njit(cache=True, fastmath=True)
def traces(SWa, SB, SK, x, c, a, d):
    for j in range(d):
        for i in range(d):
            SWa[j, i] = a[i] * SWa[j, i] + x[j] * c[i]
            SB[j, i] = a[i] * SB[j, i] + x[j] * (1.0 - a[i])
            SK[j, i] = a[i] * SK[j, i]

for n in (32, 64):
    SWa = np.random.randn(n, n); SB = np.random.randn(n, n); SK = np.random.randn(n, n)
    x = np.random.randn(n); c = np.random.randn(n); a = np.random.rand(n)
    traces(SWa, SB, SK, x, c, a, n)
    N = 20000 if n == 32 else 8000
    t0 = P()
    for _ in range(N): traces(SWa, SB, SK, x, c, a, n)
    t = (P() - t0) / N * 1e6
    print(f"d={n}: 3xd^2 traces = {t:6.2f} us  ({t*1e3/(3*n*n):.3f} ns per elemen-3-op)")
