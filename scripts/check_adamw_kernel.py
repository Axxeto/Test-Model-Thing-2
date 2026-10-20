"""Uji kesetaraan numerik adamw_step (njit) vs urutan numpy jalur fallback.

Membangun ulang PERSIS urutan operasi fast_learner.slow_update (jalur numpy), lalu
membandingkan theta/am/av hasil kernel. Toleransi: ~1e-12 (fastmath) -- BUKAN 1e-1.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2 import fastpath as fp

rng = np.random.default_rng(7)
N = 4096
tidx = np.arange(5)
offs = np.array([0, 800, 1600, 2400, 3200])
sizes = np.array([800, 800, 800, 800, N - 3200])
mat = np.array([1, 1, 0, 1, 0])
lrmul = np.array([1.0, 1.0, 1.0, 0.1, 0.1])
clip = 1e9      # matikan clip untuk uji murni AdamW dulu
wd = 0.02
lr = 5e-3
b1, b2 = 0.9, 0.999
t_adam = 37
bc1_inv = 1.0 / (1.0 - b1 ** t_adam)
bc2_inv = 1.0 / (1.0 - b2 ** t_adam)
mask = np.ones(N); mask[100:120] = 0.0

for use_mask in (False, True):
    for use_clip in (1e9, 0.5):
        gb0 = rng.standard_normal(N) * 0.05
        am0 = rng.standard_normal(N) * 0.01
        av0 = np.abs(rng.standard_normal(N)) * 0.01
        th0 = rng.standard_normal(N)
        om = mask if use_mask else None

        # ---- njit
        gb = gb0.copy(); am = am0.copy(); av = av0.copy(); th = th0.copy()
        fp.adamw_step(gb, am, av, th, np.zeros(N), lr, b1, b2, bc1_inv, bc2_inv,
                      use_clip, wd, offs, sizes, mat, lrmul, tidx, om)

        # ---- numpy (urutan identik dgn slow_update lama)
        g = gb0.copy()
        gn = float(np.sqrt((g * g).sum()))
        if gn > use_clip:
            g *= use_clip / gn
        if om is not None:
            g *= om
        m2 = am0.copy() * b1 + g * (1 - b1)
        v2 = av0.copy() * b2 + (g * g) * (1 - b2)
        upd = (m2 / (1 - b1 ** t_adam)) / (np.sqrt(v2 / (1 - b2 ** t_adam)) + 1e-8)
        th2 = th0.copy()
        dcy = 1.0 - lr * wd
        for k in range(len(tidx)):
            o = offs[k]; sz = sizes[k]
            if mat[k]:
                th2[o:o + sz] *= dcy
            th2[o:o + sz] -= lr * lrmul[k] * upd[o:o + sz]

        print(f"mask={use_mask} clip={use_clip}: "
              f"dtheta={np.abs(th-th2).max():.3e}  "
              f"dgb={np.abs(gb-g).max():.3e}  "
              f"dam={np.abs(am-m2).max():.3e}  dav={np.abs(av-v2).max():.3e}")
