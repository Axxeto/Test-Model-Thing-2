"""Batas bawah realistis jalur fused: jumlah operasi skalar Wajib per byte.

Komponen tak-terhindarkan (semua loop eksplisit, tanpa BLAS):
  fwd : update 3 jejak (d^2 *3)                        + readout/GLU (~d^2 + 3*d*ed)
  bwd : kontraksi RTRL (d^2 *3 baca + scatter)         + salinan grad ~ sama
Ukur langsung dgn loop minimal -> perkiraan lantai B/s.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from numba import njit
P = time.perf_counter

@njit(cache=True, fastmath=True)
def one_layer_fwd_floor(SWa, SB, SK, Sba, Wa, B, Wc, Wg1, Wg2, Wo, x, h, a, cc, scr, d, ed):
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += x[j] * Wa[j, i]
        ai = 0.5 * (1.0 + np.tanh(0.5 * s))
        a[i] = ai
        t = 0.0
        for j in range(d):
            t += x[j] * B[j, i]
        cc[i] = (h[i] - t) * ai * (1.0 - ai)
        h[i] = a[i] * h[i] + (1.0 - a[i]) * t
    for j in range(d):
        for i in range(d):
            SWa[j, i] = a[i] * SWa[j, i] + x[j] * cc[i]
            SB[j, i] = a[i] * SB[j, i] + x[j] * (1.0 - a[i])
            SK[j, i] = a[i] * SK[j, i]
    for i in range(d):
        Sba[i] = a[i] * Sba[i] + cc[i]
    mh = 0.0
    for i in range(d): mh += h[i]
    mh /= d
    v = 0.0
    for i in range(d):
        t = h[i] - mh
        v += t * t
    inv = 1.0 / np.sqrt(v / d + 1e-5)
    for i in range(d):
        scr[i] = (h[i] - mh) * inv
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += scr[j] * Wc[j, i]
        scr[d + i] = s
    for i in range(ed):
        z1 = 0.0; z2 = 0.0
        for j in range(d):
            z1 += scr[d + j] * Wg1[j, i]
            z2 += scr[d + j] * Wg2[j, i]
        sg = 0.5 * (1.0 + np.tanh(0.5 * z1))
        scr[2 * d + i] = z1 * sg * z2
    for i in range(d):
        s = 0.0
        for j in range(ed):
            s += scr[2 * d + j] * Wo[j, i]
        scr[3 * d + i] = s

for d in (32, 64):
    ed = 2 * d; nl = 2
    A = [np.random.randn(d, d) for _ in range(7)]
    S = [np.random.randn(d, d) for _ in range(4)]
    x = np.random.randn(d); h = np.random.randn(d)
    a = np.random.rand(d); cc = np.random.rand(d); scr = np.zeros(4 * d + ed)
    one_layer_fwd_floor(S[0], S[1], S[2], np.random.randn(d), A[0], A[1], A[2], A[3], A[4],
                        A[5], x, h, a, cc, scr, d, ed)
    N = 20000 if d == 32 else 6000
    t0 = P()
    for _ in range(N):
        one_layer_fwd_floor(S[0], S[1], S[2], np.random.randn(d), A[0], A[1], A[2], A[3], A[4],
                            A[5], x, h, a, cc, scr, d, ed)
    t = (P() - t0) / N * 1e6
    print(f"d={d}: lantai MAJU 1 lapisan = {t:6.2f} us  -> nl=2 {2*t:6.2f} us "
          f"=> {1e6/(4*t):6.0f} B/s (maju saja, bwd belum)")
