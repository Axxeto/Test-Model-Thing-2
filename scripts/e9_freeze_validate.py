"""E9 — VALIDASI FIX KURVA-U: latih PRODUKSI dengan mixer/APM dibekukan.

E8 (kurva titik-beku) memilih T = 3 jt optimal. E9 memverifikasi bahwa jalur PRODUKSI
`StatCore(mixer_freeze_after=T).train_stream(...)` **mereproduksi persis** angka cangkok E8 —
bukan hanya "cocok di atas kertas".

Referensi E8 (suite TETAP 20x200 kB i.i.d., seed 7, tb=25, CE = bit):
  utuh (tanpa beku) : 2,4883
  beku@3jt          : 2,1456   <-- target reproduksi

Kontrol wajib (aturan proyek #7): jalankan juga `--freeze 0` untuk memastikan jalur yang sama
menghasilkan 2,4883 (bila tidak, protokol/lingkungan berubah dan angka E8 tak boleh dipakai).

Jalan:
  python scripts/e9_freeze_validate.py --ntrain 30000000 --table_bits 25 --freeze 3000000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore

REF = {0: 2.4883, 3000000: 2.1456}          # referensi E8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--freeze", type=int, default=3_000_000,
                    help="byte latih setelah mana mixer/APM dibekukan (0 = tak pernah)")
    ap.add_argument("--nwin", type=int, default=20)
    ap.add_argument("--win", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", default="results/e9_freeze_validate.jsonl")
    a = ap.parse_args()

    need = a.ntrain + a.nwin * a.win
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    if len(raw) < need:
        raise SystemExit(f"korpus kurang: {len(raw)} < {need}")
    tr = raw[:a.ntrain]
    test_region = raw[a.ntrain:]
    rng = np.random.default_rng(a.seed)
    starts = rng.choice(len(test_region) - a.win, size=a.nwin, replace=False)
    windows = [test_region[s:s + a.win] for s in starts]

    orders = tuple(int(x) for x in a.orders.split(","))
    m = StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                 cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True,
                 mixer_freeze_after=a.freeze)
    print(f"[E9] tb={a.table_bits} latih {a.ntrain} byte · mixer_freeze_after={a.freeze}",
          flush=True)
    t0 = time.perf_counter()
    m.train_stream(tr)                       # satu panggilan: split bebas/beku terjadi di dalam
    el = time.perf_counter() - t0
    print(f"  latih selesai {el:.0f}s ({a.ntrain/el:.0f} B/s)", flush=True)

    per = [float(np.nanmean(m.eval_stream_fast(w))) for w in windows]
    bpb = float(np.mean(per))
    sd = float(np.std(per))
    se = sd / max(a.nwin, 1) ** 0.5
    ref = REF.get(a.freeze)
    print(f"\n== HASIL E9 ==", flush=True)
    print(f"  BPB (suite TETAP {a.nwin}x{a.win}B) = {bpb:.4f} bit  (sd {sd:.4f}, SE {se:.4f})",
          flush=True)
    if ref is not None:
        d = bpb - ref
        ok = abs(d) < 0.02
        print(f"  referensi E8 (cangkok) = {ref:.4f} -> selisih {d:+.4f} "
              f"({'COCOK (implementasi terbukti)' if ok else 'TIDAK COCOK — periksa!'})",
              flush=True)

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, table_bits=a.table_bits,
                                freeze=a.freeze, nwin=a.nwin, win=a.win, seed=a.seed,
                                bpb=bpb, sd=sd, se=se, ref_e8=ref,
                                delta_vs_ref=(None if ref is None else bpb - ref))) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
