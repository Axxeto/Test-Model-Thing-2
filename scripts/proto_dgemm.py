"""Berapa biaya loop eksplisit d x d di numba vs np.dot (BLAS) utk d=32/64?
Untuk melihat apakah kernel sudah di lantai loop."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from numba import njit
P = time.perf_counter

@njit(cache=True, fastmath=True)
def mv_loop(A, x, n):
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        out[i] = s
    return out

@njit(cache=True, fastmath=True)
def mm_loop(A, B, n):
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

for n in (32, 64):
    A = np.random.randn(n, n); x = np.random.randn(n); B = np.random.randn(n, n)
    mv_loop(A, x, n); mm_loop(A, B, n)
    N = 20000 if n == 32 else 5000
    t0 = P()
    for _ in range(N): mv_loop(A, x, n)
    t_mv = (P() - t0) / N * 1e6
    t0 = P()
    for _ in range(N // 10): mm_loop(A, B, n)
    t_mm = (P() - t0) / (N // 10) * 1e6
    t0 = P()
    for _ in range(N): A @ x
    t_mvb = (P() - t0) / N * 1e6
    t0 = P()
    for _ in range(N // 10): A @ B
    t_mmb = (P() - t0) / (N // 10) * 1e6
    ns = 1e9
    print(f"d={n}: mv loop {t_mv:6.2f} us ({t_mv*ns/n**2:.2f} ns/FLOP)  BLAS {t_mvb:6.2f} us")
    print(f"      mm loop {t_mm:7.2f} us ({t_mm*ns/n**3:.2f} ns/FLOP)  BLAS {t_mmb:7.2f} us")
