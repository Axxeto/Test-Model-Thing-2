"""Cari akar kurva-U: mengapa BPB-uji MEMBURUK setelah tabel jenuh?

Sudah DIBUANG secara empiris (jangan ulangi): ukuran tabel, count_cap, lebar checksum,
apm_lr, korelasi slot-checksum, perbedaan topik korpus. Lihat results/diag_*.jsonl.

Eksperimen di sini menjawab tiga pertanyaan yang tersisa:

A. MULTI-EPOCH. Kalau model belajar dengan benar, epoch ke-2 seharusnya MEMPERBAIKI BPB-uji
   (statistik diperkuat). Kalau malah memburuk, kerusakannya bukan "data terlalu banyak"
   melainkan mekanisme penyimpanan/adaptasi.

B. LOKAL vs GLOBAL. Latih pada EKOR data (15 jt byte terakhir) dengan ukuran sampel SAMA
   seperti eksperimen kurva-U. Kalau BPB-uji bagus di sini (uji tepat setelah blok latih),
   berarti masalahnya urutan/waktu: model "lupa" yang lama ketika terus diperbarui.

C. LR MIXER TANPA PELURUHAN. Rekam norma |w1|, |w2| pada tiap titik. Kalau membengkak
   tanpa batas -> divergensi perlahan. Juga uji lr mixer kecil (aditif 0) dan lr=0
   (mixer dibekukan setelah 10 jt) untuk memisahkan sumber.

Protokol: jendela uji selalu SETELAH blok latih (assert tak beririsan).

Jalan:
  python scripts/diag_root.py --ntrain 30000000 --ntest 20000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore

CORPUS = "data/indonesian_big.bin"


def eval_probe(m, te):
    return float(m.eval_stream(te).mean())


def run_multi_epoch(raw, ntrain, ntest, tb, orders, lr, epochs, probe_every):
    """A. Beberapa epoch pada prefix latih yang SAMA; uji setelah prefix."""
    tr = raw[:ntrain]
    te = raw[ntrain:ntrain + ntest]
    m = StatCore(orders=orders, table_bits=tb, lr=lr, use_apm=True,
                 wsel_bits=12, sparse=False, word=True)
    out = []
    n_total = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        n = 0
        nxt = probe_every
        for b in tr:
            m.step(b)
            n += 1
            n_total += 1
            if n >= nxt:
                ce = eval_probe(m, te)
                out.append(dict(exp="multi_epoch", epoch=ep + 1, n=n_total, bpb=ce,
                                w1norm=float(np.abs(m.w1).mean()),
                                w2norm=float(np.abs(m.w2).mean()),
                                fill=float(np.count_nonzero(m.tab) / m.tab.size),
                                bps=n_total / max(time.perf_counter() - t0, 1e-9)))
                nxt += probe_every
    return out


def run_local_vs_global(raw, total, ntest, tb, orders, lr, block, probe_every):
    """B. Latih pada blok BERBEDA (awal / tengah / ekor) dengan ukuran SAMA.
    Uji selalu TEPAT setelah blok latih -> memisahkan 'data banyak' dari 'urutan waktu'."""
    out = []
    starts = {"awal": 0, "tengah": (total - block - ntest) // 2, "ekor": total - block - ntest}
    for name, s in starts.items():
        tr = raw[s:s + block]
        te = raw[s + block:s + block + ntest]
        assert s + block + ntest <= total
        m = StatCore(orders=orders, table_bits=tb, lr=lr, use_apm=True,
                     wsel_bits=12, sparse=False, word=True)
        n = 0
        nxt = probe_every
        t0 = time.perf_counter()
        for b in tr:
            m.step(b)
            n += 1
            if n >= nxt:
                out.append(dict(exp="localvsglobal", where=name, start=int(s), n=n,
                                bpb=eval_probe(m, te),
                                w1norm=float(np.abs(m.w1).mean()),
                                w2norm=float(np.abs(m.w2).mean()),
                                bps=n / max(time.perf_counter() - t0, 1e-9)))
                nxt += probe_every
    return out


def run_lr_variants(raw, ntrain, ntest, tb, orders, probe_every):
    """C. Varian laju belajar mixer: lr kecil, dan lr=0 setelah pemanasan."""
    tr = raw[:ntrain]
    te = raw[ntrain:ntrain + ntest]
    out = []
    for lr in (0.002, 0.0005, 0.0001):
        m = StatCore(orders=orders, table_bits=tb, lr=lr, use_apm=True,
                     wsel_bits=12, sparse=False, word=True)
        n = 0
        nxt = probe_every
        t0 = time.perf_counter()
        for b in tr:
            m.step(b)
            n += 1
            if n >= nxt:
                out.append(dict(exp="lr_variant", lr=lr, n=n, bpb=eval_probe(m, te),
                                w1norm=float(np.abs(m.w1).mean()),
                                w2norm=float(np.abs(m.w2).mean()),
                                bps=n / max(time.perf_counter() - t0, 1e-9)))
                nxt += probe_every
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30000000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--probe_every", type=int, default=3000000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--which", default="A,B,C", help="eksperimen mana yang dijalankan")
    ap.add_argument("--jsonl", default="results/diag_root.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    total = raw.size
    which = set(x.strip().upper() for x in a.which.split(","))
    os.makedirs("results", exist_ok=True)
    fh = open(a.jsonl, "a", encoding="utf-8")

    def emit(rows, title):
        print(f"\n=== {title} ===", flush=True)
        if not rows:
            print("  (tidak ada)", flush=True)
            return
        keys = [k for k in rows[0] if k not in ("exp",)]
        print("  " + " ".join(f"{k:>12}" for k in keys), flush=True)
        for r in rows:
            print("  " + " ".join(
                (f"{r[k]:>12.4f}" if isinstance(r.get(k), float) else f"{str(r.get(k)):>12}")
                for k in keys), flush=True)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()

    if "A" in which:
        rows = run_multi_epoch(raw, a.ntrain, a.ntest, a.table_bits, orders, 0.002,
                               a.epochs, a.probe_every)
        emit(rows, f"A. MULTI-EPOCH ({a.epochs} epoch pada prefix {a.ntrain//10**6} jt)")

    if "B" in which:
        rows = run_local_vs_global(raw, total, a.ntest, a.table_bits, orders, 0.002,
                                   a.ntrain, a.probe_every)
        emit(rows, f"B. LOKAL vs GLOBAL (blok {a.ntrain//10**6} jt, uji tepat setelah blok)")

    if "C" in which:
        rows = run_lr_variants(raw, a.ntrain, a.ntest, a.table_bits, orders, a.probe_every)
        emit(rows, "C. VARIAN lr MIXER")

    fh.close()
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
