"""Matriks V0-V5 CEPAT di testbed multi-epoch 1MB x 20 (Fase 1, ~15 menit).

Testbed ini mereproduksi drift mixer (BPB-uji +0,5, max|w1| -> ribuan) dalam
~8 menit/varian, vs 30MB single-pass. Pohon keputusan §6:
  V1 gerbang saja (diagnostik, bukan produksi) dilewati di sini karena butuh
  edit kernel tambahan; M1 diuji via sat_frac + ownCE dari diagnose().
  V0 legacy | V2 mode1 polos | V3 +jadwal | V4 +L2 | V5 +s2_clip=8

Jalan: python scripts/vmatrix_fast.py  (hasil -> results/vmatrix_fast.jsonl)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()

N = 1_000_000
EPOCHS = 20
TB = 22
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)

VARIANTS = [
    ("V0-legacy", dict(mixer_mode=0)),
    ("V2-mode1", dict(mixer_mode=1)),
    ("V3+sched", dict(mixer_mode=1, mix_lr_nhalf=50000, mix_lr_floor=0.02)),
    ("V4+L2", dict(mixer_mode=1, mix_lr_nhalf=50000, mix_lr_floor=0.02,
                   mix_l2=1e-4)),
    ("V5+clip8", dict(mixer_mode=1, mix_lr_nhalf=50000, mix_lr_floor=0.02,
                      mix_l2=1e-4, s2_clip=8.0)),
]


def main():
    raw = np.frombuffer(open("data/indonesian_big.bin", "rb").read(N + 20000),
                        dtype=np.uint8)
    tr, te = raw[:N].copy(), raw[N:N + 20000].copy()
    print(f"testbed: latih {N}B x {EPOCHS} epoch, uji tetap {len(te)}B, tb={TB}",
          flush=True)
    with open("results/vmatrix_fast.jsonl", "a", encoding="utf-8") as f:
        for name, kw in VARIANTS:
            m = StatCore(orders=ORDERS, table_bits=TB, wsel_bits=12, cks_bits=24,
                         match_fix=True, ctx_fix=True, use_apm=True, **kw)
            t0 = time.perf_counter()
            row0 = None
            for ep in range(1, EPOCHS + 1):
                m.train_stream(tr)
                ctx = m.stream_ctx()
                bpb = float(np.nanmean(m.eval_stream_fast(te)))
                m.restore_ctx(ctx)
                if ep in (1, EPOCHS):
                    d = m.diagnose(te[:10000])
                    row = dict(var=name, ep=ep, bpb=round(bpb, 4),
                               w1max=round(d["w1max"], 2),
                               sat=[round(s, 3) for s in d["sat_frac"]],
                               ownCE=[round(v * 8, 3) for v in d["own_ce"]],
                               w2neg=round(d["w2neg_frac"], 4))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    if ep == 1:
                        row0 = row
                    print(f"  {name} ep{ep:2d}: BPB {bpb:.4f} max|w1| {d['w1max']:9.1f} "
                          f"sat {row['sat']} ownCE {row['ownCE']}", flush=True)
            dt = time.perf_counter() - t0
            print(f"[{name}] {dt:.0f}s  ΔBPB {row['bpb']-row0['bpb']:+.4f} "
                  f"(ep1 {row0['bpb']:.4f} -> ep20 {row['bpb']:.4f})", flush=True)


if __name__ == "__main__":
    main()
