"""E2 — INSTRUMENTASI kurva-U: hit-rate per konteks, norma bobot mixer, isi tabel.

Audit 2026-10-06 §3.4 E2. Tujuan: memutus H-A (mesin tabel) vs H-B (drift mixer/APM)
dengan mengukur INDIKATOR INTERNAL tiap ~1 jt byte, TANPA mengubah hasil.

Yang diukur tiap interval:
  * fraksi lookup "KENA" (checksum cocok) per konteks/orde   -> bila turun sepanjang latih => H-A
  * rata-rata |stretch| tiap konteks                          -> sinyal order-tinggi kolaps?
  * max|w1|, mean|w1|, mean|w2|                               -> bila naik terus => H-B (drift)
  * isi tabel (slot terpakai / total)                         -> jenuh?
  * akurasi match (mpred benar) & cakupan (mpred>=0)          -> H-E

Paritas: evaluasi memakai train=False dan konteks tetap berjalan (aturan #1 proyek).

Jalan:
  python scripts/e2_instrument.py --corpus data/indonesian_big.bin --ntrain 30000000 --table_bits 25
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--ntest", type=int, default=200_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--interval", type=int, default=1_000_000)
    ap.add_argument("--probe_bytes", type=int, default=20_000)
    ap.add_argument("--jsonl", default="results/e2_instrument.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    need = a.ntrain + a.ntest
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:]

    m = StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                 cks_bits=a.cks_bits, match_fix=True, ctx_fix=True)
    print(f"[E2] orders={orders} tb={a.table_bits} interval={a.interval} "
          f"probe={a.probe_bytes}B (jalur TERKOMPILASI)", flush=True)

    # ⚡ Jalur CEPAT: seluruh latih + probe di dalam satu kernel numba (dulu loop Python
    # per byte + probe per byte = ~6,6 kB/s; kini ~170 kB/s pada tb=18, naik seiring tb).
    t0 = time.perf_counter()
    (hr_tab, av_tab, idx, w1mx, w1mn, w2mn, fill) = m.instrument_stream(
        tr, interval=a.interval, probe_bytes=a.probe_bytes)
    el = time.perf_counter() - t0

    logs = []
    for k in range(len(idx)):
        end = int(idx[k])
        hr = hr_tab[k]
        av = av_tab[k]
        rec = dict(
            n=end,
            table_fill=round(float(fill[k]), 5),
            w1_max=float(w1mx[k]),
            w1_mean=float(w1mn[k]),
            w2_mean=float(w2mn[k]),
            hit_rate=[round(float(x), 4) for x in hr],
            abs_stretch=[round(float(x), 4) for x in av],
            bps=round(end / max(el, 1e-9), 0),
        )
        logs.append(rec)
        print(f"  @{end/1e6:5.2f}jt fill={rec['table_fill']:.3f} |w1|mx={rec['w1_max']:.2f} "
              f"|w2|mn={rec['w2_mean']:.4f} "
              f"hit[0,1,3,5,7,9]={hr[0]:.2f},{hr[1]:.2f},{hr[3]:.2f},{hr[5]:.2f},"
              f"{hr[7]:.2f},{hr[9]:.2f} |s|[0,3,5,7,9]={av[0]:.2f},{av[3]:.2f},{av[5]:.2f},"
              f"{av[7]:.2f},{av[9]:.2f}", flush=True)

    bpb = float(np.nanmean(m.eval_stream_fast(te)))
    print(f"\nBPB uji tetap ({len(te)}B): {bpb:.4f}   (laju latih+probe {a.ntrain/el:.0f} B/s)")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, table_bits=a.table_bits,
                                orders=list(orders), bpb=bpb, logs=logs)) + "\n")
    print(f"ditulis -> {a.jsonl}")

    # ringkasan tren
    if len(logs) >= 2:
        print("\n== TREN ==")
        first, last = logs[0], logs[-1]
        print(f"  table_fill : {first['table_fill']:.3f} -> {last['table_fill']:.3f}")
        print(f"  |w1|max    : {first['w1_max']:.2f} -> {last['w1_max']:.2f}")
        print(f"  |w1|mean   : {first['w1_mean']:.4f} -> {last['w1_mean']:.4f}")
        print(f"  |w2|mean   : {first['w2_mean']:.4f} -> {last['w2_mean']:.4f}")
        hr_f, hr_l = np.array(first['hit_rate']), np.array(last['hit_rate'])
        print(f"  hit orde rendah (idx0-3): {hr_f[:4].round(3)} -> {hr_l[:4].round(3)}")
        print(f"  hit orde tinggi (idx4+): {hr_f[4:].round(3)} -> {hr_l[4:].round(3)}")


if __name__ == "__main__":
    main()
