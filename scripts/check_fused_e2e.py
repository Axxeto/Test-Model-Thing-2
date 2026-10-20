"""Uji end-to-end: jalur KERNEL TERPADU vs jalur lama untuk 2 byte, bandingkan arena gradien.

Protocol: dua FastLearner dengan model IDENTIK (seed sama), satu pakai use_fused_kernel,
satu jalur lama. Jalankan 2 byte yang sama, lalu `slow_update` dipaksa agar gradien
terealisasi -- bandingkan theta setelah SATU slow_update. Beda harus ~1e-15.

CATATAN: jangan jalankan lintasan panjang (chaos online menggeser hasil).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner


def make(seed, fused):
    cfg = Config.best(d=16, n_layers=2, vocab=64, d_mem=8, use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=seed,
                  use_stop=False)
    L = FastLearner(m, o)
    return cfg, m, L


def run(L, stream, data):
    ce = []
    for b in data:
        c = L.step(stream, b, train=True) if not getattr(L, "_fused", False) else L.step_fused(stream, b, train=True)
        ce.append(c)
    return ce


def main():
    data = [3, 17, 42, 5]
    # ---- jalur lama
    cfgA, mA, LA = make(0, False)
    sA = Stream(mA, online=True)
    # ---- jalur fused
    cfgB, mB, LB = make(0, True)
    sB = Stream(mB, online=True)

    # pastikan bobot awal identik
    def maxdiff(LA_, MA, LB_, MB):
        w = 0.0
        for (nA, pA), (nB, pB) in zip(MA.params.items(), MB.params.items()):
            a = np.asarray(pA.data); b = np.asarray(pB.data)
            if a.size == 0:
                continue
            w = max(w, float(np.abs(a - b).max()))
        return w
    d0 = maxdiff(LA, mA, LB, mB)
    print("bobot awal beda:", d0)

    ceA = run(LA, sA, data[:2])
    ceB = run(LB, sB, data[:2])
    print("CE lama :", [None if c is None else round(float(c), 6) for c in ceA])
    print("CE fused:", [None if c is None else round(float(c), 6) for c in ceB])

    # ---- bandingkan akumulasi gradien 'byte' (setelah 2 byte, sebelum slow_update)
    gA = LA.gacc.get("byte")
    gB = LB.gacc.get("byte")
    worst, wn = 0.0, ""
    for nm in LA.offs:
        o, shp = LA.offs[nm]
        sz = int(np.prod(shp)) if shp else 1
        a = gA[o:o + sz] if gA is not None else np.zeros(sz)
        b = gB[o:o + sz] if gB is not None else np.zeros(sz)
        e = float(np.abs(a - b).max()) if sz else 0.0
        if e > worst:
            worst, wn = e, nm
    print(f"grad 'byte' worst = {worst:.3e} ({wn}) -> {'OK' if worst < 1e-12 else 'BEDA'}")
    if gA is not None and gB is not None:
        print("  arena diff total:", float(np.abs(gA - gB).max()))

    # ---- paksa slow_update & bandingkan theta
    LA.slow_update(sA)
    LB.slow_update(sB)
    dt = maxdiff(LA, mA, LB, mB)
    print(f"theta setelah 1 slow_update: beda maks = {dt:.3e} -> "
          f"{'OK' if dt < 1e-12 else 'BEDA'}")


if __name__ == "__main__":
    main()
