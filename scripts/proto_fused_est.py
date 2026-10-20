"""Estimasi batas atas kernel TERPADU: satu njit per byte, tanpa dispatch berulang.

Mengukur biaya inti komputasi 1 byte (2 lapisan: fwd+backward RTRL, + kepala) BILA
semuanya di dalam SATU panggilan njit (state di array datar, tanpa alokasi intermediate
besar). Ini batas bawah realistis utk arsitektur terpadu.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from numba import njit

d, dm, ed = 32, 8, 64
NL = 2

@njit(cache=True, fastmath=True)
def fused_byte(x, h1, h2, SWa1, SB1, SK1, Sba1, SWa2, SB2, SK2, Sba2,
               Wa1, ba1, B1, K1, Wc1, Wq1, Wm1, Wg1_1, Wg2_1, Wo1, Wa2, ba2, B2, K2,
               Wc2, Wq2, Wm2, Wg1_2, Wg2_2, Wo2, M1, M2, g, W_out, scr):
    # lapisan 1: rekurensi + jejak (loop eksplisit)
    a = scr[0:d]; u = scr[d:2*d]; cc = scr[2*d:3*d]
    for i in range(d):
        s = ba1[i]
        for j in range(d):
            s += x[j] * Wa1[j, i]
        ai = 0.5 * (1.0 + np.tanh(0.5 * s))
        a[i] = ai
        t = 0.0
        for j in range(d):
            t += x[j] * B1[j, i]
        u[i] = t
        cc[i] = (h1[i] - t) * ai * (1.0 - ai)
    h1n = scr[3*d:4*d]
    for i in range(d):
        h1n[i] = a[i] * h1[i] + (1.0 - a[i]) * u[i]
    for j in range(d):
        for i in range(d):
            SWa1[j, i] = a[i] * SWa1[j, i] + x[j] * cc[i]
            SB1[j, i] = a[i] * SB1[j, i] + x[j] * (1.0 - a[i])
            SK1[j, i] = a[i] * SK1[j, i]
    for i in range(d):
        Sba1[i] = a[i] * Sba1[i] + cc[i]
    # readout LN + Wc
    mh = 0.0
    for i in range(d): mh += h1n[i]
    mh /= d
    hn = scr[4*d:5*d]
    var = 0.0
    for i in range(d):
        t = h1n[i] - mh; var += t*t
    inv = 1.0/np.sqrt(var/d + 1e-5)
    for i in range(d): hn[i] = (h1n[i]-mh)*inv
    c0 = scr[5*d:6*d]
    for i in range(d): c0[i] = 0.0
    for j in range(d):
        hj = hn[j]
        for i in range(d): c0[i] += hj*Wc1[j, i]
    # GLU
    z1 = scr[6*d:6*d+ed]; z2 = scr[6*d+ed:6*d+2*ed]
    for i in range(ed): z1[i] = 0.0; z2[i] = 0.0
    for j in range(d):
        cj = c0[j]
        for i in range(ed):
            z1[i] += cj*Wg1_1[j, i]; z2[i] += cj*Wg2_1[j, i]
    y = scr[6*d+2*ed:6*d+2*ed+d]
    for i in range(ed):
        sg = 0.5*(1.0+np.tanh(0.5*z1[i])); y[i] = z1[i]*sg*z2[i]
    for i in range(d):
        s = 0.0
        for j in range(ed): s += Wo1[i, j]*y[j]
        h2[i] += s
    # lapisan 2 sama (ringkas: hanya fwd rekurensi)
    a2 = scr[0:d]; u2 = scr[d:2*d]
    for i in range(d):
        s = ba2[i]
        for j in range(d): s += h2[j]*Wa2[j, i]
        ai = 0.5*(1.0+np.tanh(0.5*s)); a2[i]=ai
        t=0.0
        for j in range(d): t += h2[j]*B2[j, i]
        u2[i]=t
    h2n = scr[3*d:4*d]
    for i in range(d): h2n[i] = a2[i]*h2[i] + (1.0-a2[i])*u2[i]
    for j in range(d):
        for i in range(d):
            SWa2[j,i] = a2[i]*SWa2[j,i] + h2[j]*(h2[i]-u2[i])*a2[i]*(1.0-a2[i])
    # kepala linear + gradien (V=256)
    p = scr[4*d:5*d]
    nrm = 0.0
    for i in range(d): nrm += h2n[i]*h2n[i]
    nrm = np.sqrt(nrm + 1e-8)
    for i in range(d): p[i] = h2n[i]/nrm
    V = W_out.shape[0]
    lg = scr[6*d:6*d+V]
    for i in range(V):
        s = 0.0
        for j in range(d): s += W_out[i, j]*p[j]
        lg[i] = s
    mx = lg[0]
    for i in range(V):
        if lg[i] > mx: mx = lg[i]
    se = 0.0
    for i in range(V): se += np.exp(lg[i]-mx)
    for i in range(V):
        gi = np.exp(lg[i]-mx)/se
        for j in range(d):
            g[i*d + j] += gi*p[j]
    return 0.0


def main():
    rng = np.random.default_rng(0)
    z = lambda *s: np.zeros(s)
    x = rng.standard_normal(d); h1 = rng.standard_normal(d); h2 = rng.standard_normal(d)
    Wd = lambda: rng.standard_normal((d, d)) * 0.1
    Wa1, B1, Wa2, B2, Wc1, Wc2 = Wd(), Wd(), Wd(), Wd(), Wd(), Wd()
    K1, K2 = Wd() * 0.1, Wd() * 0.1
    SWa1, SB1, SK1, SWa2, SB2, SK2 = Wd() * 0.01, Wd() * 0.01, Wd() * 0.01, Wd() * 0.01, Wd() * 0.01, Wd() * 0.01
    ba1 = z(d); ba2 = z(d)
    Wq1 = Wd()[:, :dm]; Wm1 = rng.standard_normal((dm, d)); Wg1_1 = Wd()[:, :ed]; Wg2_1 = Wd()[:, :ed]
    Wo1 = rng.standard_normal((d, ed)); Wq2 = Wd()[:, :dm]; Wm2 = rng.standard_normal((dm, d))
    Wg1_2 = Wd()[:, :ed]; Wg2_2 = Wd()[:, :ed]; Wo2 = rng.standard_normal((d, ed))
    M1 = z(dm, dm); M2 = z(dm, dm)
    g = z(256 * d); W_out = rng.standard_normal((256, d)) * 0.1
    scr = z(4096)
    args = (x, h1, h2, SWa1, SB1, SK1, ba1, SWa2, SB2, SK2, ba2,
            Wa1, ba1, B1, K1, Wc1, Wq1, Wm1, Wg1_1, Wg2_1, Wo1, Wa2, ba2, B2, K2,
            Wc2, Wq2, Wm2, Wg1_2, Wg2_2, Wo2, M1, M2, g, W_out, scr)
    fused_byte(*args)
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        fused_byte(*args)
    dt = (time.perf_counter() - t0) / N
    print(f"kernel TERPADU (2 lapisan + kepala V=256, 1 panggilan): {1e6*dt:7.2f} us/byte")
    print(f"  batas atas throughput (hanya kernel): {1/dt:9.0f} B/s")
    print(f"  + dispatch ~2 us -> {(1e6*dt+2):7.2f} us  => {1e6/(1e6*dt+2):9.0f} B/s")


if __name__ == "__main__":
    main()
