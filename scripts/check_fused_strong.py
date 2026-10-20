"""Uji e2e KUAT: jalur fused vs jalur lama dgn BANYAK slow_update.

Perbandingan dilakukan byte-per-byte PADA theta yang sama: setiap byte, panggil backward
di kedua jalur pada state identik, lalu bandingkan arena gradien 'byte' SEBELUM SU merubah
theta. Karena aktivasi-basi hanya muncul pada backward pertama pasca-SU, loop ini menembak
tepat kasus yang dulu gagal.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2 import fastpath as fp


def make(seed, fused, se, nl, d, dm, V):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=dm, use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False, slow_every=se)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=seed, use_stop=False)
    return cfg, m, FastLearner(m, o)


def main():
    rng = np.random.default_rng(7)
    for (nl, d, dm, V, se, n) in [(1, 10, 4, 40, 4, 24), (2, 10, 4, 40, 4, 24),
                                  (2, 12, 6, 48, 3, 30), (3, 10, 4, 40, 2, 30),
                                  (2, 10, 4, 40, 16, 40)]:
        data = [int(x) for x in rng.integers(0, V, size=n)]
        cfgA, mA, LA = make(0, False, se, nl, d, dm, V)
        cfgB, mB, LB = make(0, True, se, nl, d, dm, V)
        sA, sB = Stream(mA, online=True), Stream(mB, online=True)
        worst_g, worst_nm, worst_step = 0.0, "", -1
        ce_ok = True
        for i, b in enumerate(data):
            ca = LA.step(sA, b, train=True)
            cb = LB.step_fused(sB, b, train=True)
            if (ca is None) != (cb is None):
                ce_ok = False
            elif ca is not None and abs(ca - cb) > 1e-9:
                ce_ok = False
            # bandingkan arena gradien byte ini (setelah langkah ini, sebelum SU berikutnya)
            gA = LA.gacc.get("byte"); gB = LB.gacc.get("byte")
            if gA is not None and gB is not None:
                e = float(np.abs(gA - gB).max())
                if e > worst_g:
                    worst_g, worst_step = e, i
                    wn = max(LA.offs, key=lambda nm: float(np.abs(
                        gA[LA.offs[nm][0]:LA.offs[nm][0] + (int(np.prod(LA.offs[nm][1])) if LA.offs[nm][1] else 1)] -
                        gB[LA.offs[nm][0]:LA.offs[nm][0] + (int(np.prod(LA.offs[nm][1])) if LA.offs[nm][1] else 1)]).max()))
                    worst_nm = wn
        # theta akhir
        dt = 0.0
        for (nA, pA), (nB, pB) in zip(mA.params.items(), mB.params.items()):
            a = np.asarray(pA.data); bq = np.asarray(pB.data)
            if a.size == 0:
                continue
            dt = max(dt, float(np.abs(a - bq).max()))
        status = "OK" if (worst_g < 1e-12 and dt < 1e-12 and ce_ok) else "BEDA"
        print(f"nl={nl} d={d} se={se} n={n}: grad byte worst {worst_g:.3e} @byte{worst_step}({worst_nm})  "
              f"theta {dt:.3e}  CE {'ok' if ce_ok else 'BEDA'}  -> {status}")


if __name__ == "__main__":
    main()
