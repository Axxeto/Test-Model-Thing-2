"""Root-cause kurva-U dengan protokol TERNORMALISASI (menjawab kritik analisis).

MASALAH METODOLOGI YANG DIPERBAIKI DI SINI:
  1. Jendela uji 20 kB itu BISING (±0,05). Di sini uji = dokumen acak terpisah,
     tersebar merata, >=1 MB (uji variansi jauh lebih kecil).
  2. Perbandingan lintas-skrip sebelumnya terkonfound oleh `cks_bits` berbeda dan
     jadwal probe berbeda. Di sini SATU skrip, SATU jadwal, `cks_bits` EKSPLISIT.
  3. Kesimpulan "dibuang" sebelumnya diambil pada rezim tak-jenuh. Di sini probe
     dijalankan sampai tabel benar-benar jenuh.

RANCANGAN (semua laporan BPB memakai set uji DOKUMEN ACAK yang sama):
  * Dokumen diacak (seed tetap), 1% ditahan sebagai uji.
  * Dua urutan latih: (a) ASLI (dokumen berurutan), (b) ACAK.
    Jika kurva-U hilang pada urutan acak -> penyebabnya urutan/drift (hip. A).
    Jika tetap ada -> penyebabnya tabel/desain (hip. B).

RUANG LINGKUP: 30 jt byte, tb=25, cks_bits eksplisit 8 dan 24.

Jalan:
  python scripts/diag_normalized.py --ntrain 30000000
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
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)


def build_splits(raw, nbytes, test_frac=0.01, seed=0):
    """Bagi jadi dokumen, acak, tahan ~test_frac sebagai uji tersebar merata.
    Mengembalikan (urutan_asli_train, urutan_acak_train, tekon_uji_bytes)."""
    data = bytes(raw[:nbytes])
    docs = [d for d in data.split(b"\n\n") if len(d) > 20]
    idx = np.arange(len(docs))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(docs))
    n_test = max(1, int(len(docs) * test_frac))
    test_set = set(perm[:n_test].tolist())

    # teks uji: dokumen uji digabung (urutan tak penting, hanya datanya)
    test_txt = b"\n\n".join(docs[i] for i in sorted(test_set))
    # (a) urutan ASLI: dokumen latih dalam urutan indeks asli
    train_orig = b"\n\n".join(docs[i] for i in idx if i not in test_set)
    # (b) urutan ACAK: dokumen latih diacak (seed tetap)
    train_shuf = b"\n\n".join(docs[i] for i in perm if i not in test_set)
    return (np.frombuffer(train_orig, dtype=np.uint8),
            np.frombuffer(train_shuf, dtype=np.uint8),
            np.frombuffer(test_txt, dtype=np.uint8), len(docs), n_test)


def train_and_probe(m, tr, te, probes, tag, fh):
    n = 0
    pi = 0
    t0 = time.perf_counter()
    while pi < len(probes):
        # latih sampai probe berikutnya
        target = min(probes[pi], len(tr))
        while n < target:
            m.step(int(tr[n]))
            n += 1
        bpb = float(m.eval_stream(te).mean())
        fill = float(np.count_nonzero(m.tab) / m.tab.size)
        rec = dict(tag=tag, n=n, bpb=bpb, table_fill=fill,
                   bps=n / max(time.perf_counter() - t0, 1e-9))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"    {tag:<18} n={n:>10,} BPB-uji={bpb:.4f} isi={fill*100:5.1f}% "
              f"({rec['bps']:.0f} B/s)", flush=True)
        pi += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30000000)
    ap.add_argument("--cks_bits", default="8,24")
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--probe_every", type=int, default=5000000)
    ap.add_argument("--jsonl", default="results/diag_normalized.jsonl")
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = f.read(a.ntrain + 2_000_000)
    tr_orig, tr_shuf, te, ndocs, ntest = build_splits(raw, a.ntrain)
    print(f"dokumen={ndocs} | uji={ntest} dok ({len(te)} B, tersebar merata) | "
          f"latih-asli={len(tr_orig)} B | latih-acak={len(tr_shuf)} B", flush=True)
    print(f"probe tiap {a.probe_every:,} B | tb={a.table_bits} | cks_bits={a.cks_bits}\n",
          flush=True)

    os.makedirs("results", exist_ok=True)
    fh = open(a.jsonl, "a", encoding="utf-8")

    for cks in [int(x) for x in a.cks_bits.split(",")]:
        probes = list(range(a.probe_every, a.ntrain + 1, a.probe_every))
        for name, tr in (("urutan_asli", tr_orig), ("urutan_acak", tr_shuf)):
            tag = f"cks{cks}/{name}"
            print(f"  == cks_bits={cks} · {name} ==", flush=True)
            m = StatCore(orders=ORDERS, table_bits=a.table_bits, lr=0.002, use_apm=True,
                         wsel_bits=12, sparse=False, word=True, cks_bits=cks)
            train_and_probe(m, tr, te, probes, tag, fh)
            print("", flush=True)

    fh.close()
    print(f"ditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
