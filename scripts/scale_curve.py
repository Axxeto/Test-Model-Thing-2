"""G4 — kurva skala Tingkat A: BPB tahan vs ukuran data latih (1M → 10M byte).

Protokol sama dengan bench_stat_core: latih pada data[:n], nilai pada jendela uji
TAK TERLIHAT data[n:n+ntest]. Tujuan: verifikasi kurva TURUN MONOTON tanpa plateau (G4),
dan menaruh G1 WikiText pada skala yang dokumen minta (≥1M).

Korpus lokal terbatas (WikiText 1,12 MB · Indonesia 965 kB). **JENDELA UJI TETAP**: selalu
8000 byte TERAKHIR korpus, dipakai untuk SEMUA ukuran latih; ukuran latih dipangkas agar tak
menyentuh jendela uji. Ini wajib — versi pertama memakai `raw[n:n+ntest]` sehingga tiap baris
kurva dinilai di jendela berbeda dan angkanya tak bisa dibandingkan (goyangan ±0,09 bit palsu).

Pakai: python scripts/scale_curve.py --corpus wikitext --sizes 250000,500000,1000000
       python scripts/scale_curve.py --corpus wikitext --sizes 250000,500000,1000000 --table_bits 26
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.stat_core import StatCore

CORPORA = {"wikitext": "wikitext", "indonesian": "data/indonesian_1m_pure.txt"}


def one(kind, ntrain, ntest, table_bits, lr, orders, apm, match_minlen, wsel_bits,
        sparse, word, warm=4000):
    raw = np.frombuffer(load(CORPORA[kind]), dtype=np.uint8)
    total = raw.size
    # JENDELA UJI TETAP (uji-tahan tunggal): selalu 8000 byte terakhir korpus, dipakai untuk
    # SEMUA ukuran latih. Tanpa ini, tiap baris kurva dinilai pada jendela berbeda → angkanya
    # TIDAK BISA dibandingkan (jebakan protokol ke-4; lihat catatan di kepala berkas).
    te = raw[total - ntest:]
    avail = total - ntest
    ntrain = min(ntrain, avail)
    tr = raw[:ntrain]
    m = StatCore(orders=orders, table_bits=table_bits, lr=lr, use_apm=apm,
                 match_minlen=match_minlen, match_bits=20, match_conf=2.0,
                 wsel_bits=wsel_bits, sparse=sparse, word=word)
    warm = min(warm, ntrain)
    for b in tr[:warm]:
        m.step(b)
    rest = tr[warm:]
    t0 = time.perf_counter()
    for b in rest:
        m.step(b)
    dt = time.perf_counter() - t0
    bps = (len(rest) / dt) if dt > 0 else float("nan")
    ce = float(m.eval_stream(te).mean())
    return ce, bps, len(te), int(m.n_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--sizes", default="250000,500000,1000000")
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--table_bits", type=int, default=22)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--wsel_bits", type=int, default=8)
    ap.add_argument("--no_apm", action="store_true")
    ap.add_argument("--no_sparse", action="store_true", default=True,
                    help="konteks jarang MERUGI (+0,038 bit) → default MATI (produksi)")
    ap.add_argument("--sparse", dest="no_sparse", action="store_false",
                    help="nyalakan konteks jarang (ablasi)")
    ap.add_argument("--no_word", action="store_true")
    ap.add_argument("--match_minlen", type=int, default=6)
    ap.add_argument("--seed_reps", type=int, default=1)
    ap.add_argument("--out", default="results/scale_curve.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    sizes = [int(x) for x in a.sizes.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(f"G4 kurva skala · korpus={a.corpus} · table_bits={a.table_bits} · "
          f"wsel={a.wsel_bits} · test={a.ntest}")
    print(f"{'ntrain':>10}{'BPB':>10}{'B/s':>10}{'n uji':>8}{'detik':>9}")
    print("-" * 47)
    rows = []
    prev = None
    for n in sizes:
        t0 = time.perf_counter()
        ce, bps, nu, steps = one(a.corpus, n, a.ntest, a.table_bits, a.lr, orders,
                                 not a.no_apm, a.match_minlen, a.wsel_bits,
                                 not a.no_sparse, not a.no_word)
        el = time.perf_counter() - t0
        d = "" if prev is None else f"  (Δ vs prev {ce - prev:+.4f})"
        print(f"{n:>10}{ce:>10.4f}{bps:>10.0f}{nu:>8}{el:>9.1f}{d}", flush=True)
        rows.append(dict(corpus=a.corpus, ntrain=n, bpb=ce, bps=bps, ntest=nu,
                         table_bits=a.table_bits, wsel_bits=a.wsel_bits, orders=list(orders),
                         apm=not a.no_apm, sparse=not a.no_sparse, word=not a.no_word))
        prev = ce
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mono = all(rows[i + 1]["bpb"] < rows[i]["bpb"] for i in range(len(rows) - 1))
    print(f"\nmonoton turun: {'YA' if mono else 'TIDAK'}")
    print(f"ditulis ke {a.out}")


if __name__ == "__main__":
    main()
