import sys, time
sys.path.insert(0, '.')
import numpy as np
from numba import njit
@njit(cache=True)
def noop(): pass
@njit(cache=True)
def noop1(a): return a[0]
@njit(cache=True)
def noop5(a,b,c,dd,e): return a[0]
@njit(cache=True)
def nooptup(t): return t[0][0]
P = time.perf_counter
arrs = [np.zeros(1) for _ in range(5)]
tup = (arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], 0, 0, 0, 0, 0, 0, 0, 0)
for name, fn, args, n in [('noop0', noop, (), 200000),
                          ('noop1', noop1, (arrs[0],), 200000),
                          ('noop5', noop5, tuple(arrs), 200000),
                          ('nooptup13', nooptup, (tup,), 200000)]:
    fn(*args)
    t0 = P()
    for _ in range(n): fn(*args)
    print(f"{name:10s} {(P()-t0)/n*1e6:.3f} us")
