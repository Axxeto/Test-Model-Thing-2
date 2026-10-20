"""EKSPERIMEN 1: CE-seg ternormalisasi per blok, dibandingkan bz2 -9 per blok.

TUJUAN (menjawab kritik analisis):
  Kalau CE kita naik di blok akhir sedangkan kesulitan bz2 blok itu TIDAK naik,
  maka kenaikan itu berasal dari MODEL (tabel/match/penghitung), bukan dari data.

CARA: satu model dilatih berurutan atas seluruh korpus. Setiap blok 1 MB:
  - ukur CE rata-rata kita pada blok itu (protokol prekuensial: prediksi byte
    berikutnya SEBELUM melihatnya; ini CE latih-in-sample tapi per blok, jadi
    menangkap DRIFT relatif),
  - ukur bz2 -9 panjang blok itu (batas atas "kesulitan intrinsik" per blok),
  - ukur rasio CE_kita / CE_bz2 sebagai "efisiensi relatif".

Kalau rasio ini ~konstan sepanjang 190 MB -> model kita melacak kesulitan data;
tidak ada degradasi. Kalau rasio NAIK -> degradasi internal model.

CATATAN PROTOKOL: CE di sini diukur pada byte yang BARU DIPELAJARI (latih),
bukan held-out. Itu DISENGAJA: kita ingin mengukur "seberapa sulit blok ini
bagi model yang sudah melihat konteks sebelumnya", bukan generalisasi. Untuk
generalisasi, lihat eksperimen 2 (uji dokumen-tak-terlihat).

Jalan:
  python scripts/diag_blocks.py --ntrain 190000000 --block 1000000
"""
import argparse
import bz2
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore

CORPUS = "data/indonesian_big.bin"
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)


def bz2_bpb(block: bytes) -> float:
    """bit/byte bz2 -9 untuk satu blok (batas atas kesulitan intrinsik blok)."""
    if not block:
        return float("nan")
    c = bz2.compress(block, 9)
    return 8.0 * len(c) / len(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=190_000_000)
    ap.add_argument("--block", type=int, default=1_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--count_cap", type=int, default=1023)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--match_fix", type=int, default=1)
    ap.add_argument("--ctx_fix", type=int, default=1)
    ap.add_argument("--jsonl", default="results/diag_blocks.jsonl")
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(a.ntrain), dtype=np.uint8)
    total = len(raw)
    print(f"korpus={total:,} B | blok={a.block:,} B | tb={a.table_bits} "
          f"count_cap={a.count_cap} cks={a.cks_bits} match_fix={a.match_fix} "
          f"ctx_fix={a.ctx_fix}", flush=True)

    m = StatCore(orders=ORDERS, table_bits=a.table_bits, lr=0.002, use_apm=True,
                 wsel_bits=12, sparse=False, word=True, cks_bits=a.cks_bits,
                 count_cap=a.count_cap,
                 match_fix=bool(a.match_fix), ctx_fix=bool(a.ctx_fix))

    os.makedirs("results", exist_ok=True)
    fh = open(a.jsonl, "a", encoding="utf-8")

    # CE berjalan per blok: akumulasi NLL/ln2 pada blok, reset tiap blok.
    n = 0
    blk = 0
    t0 = time.perf_counter()
    while n < total:
        end = min(n + a.block, total)
        # ukur CE kita pada blok [n, end): step() mengembalikan CE(bit) byte ini,
        # dihitung prekuensial lalu model langsung belajar byte tsb.
        nll = 0.0
        cnt = 0
        for i in range(n, end):
            b = int(raw[i])
            ce = m.step(b)                 # CE(bit) byte ini; None utk byte teramat awal
            if ce is not None:
                nll += ce
                cnt += 1
        ce_ours = nll / max(cnt, 1)
        block_bytes = raw[n:end].tobytes()
        ce_bz2 = bz2_bpb(block_bytes)
        rec = dict(blk=blk, n0=int(n), n1=int(end), ce_ours=ce_ours, ce_bz2=ce_bz2,
                   ratio=ce_ours / ce_bz2, table_bits=a.table_bits,
                   count_cap=a.count_cap,
                   fill=float(np.count_nonzero(m.tab) / m.tab.size),
                   bps=(end - n) / max(time.perf_counter() - t0, 1e-9))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if blk % 5 == 0 or end >= total:
            print(f"  blok {blk:>3} [{n/1e6:6.1f}–{end/1e6:6.1f} MB] "
                  f"CE={ce_ours:.4f} bz2={ce_bz2:.4f} rasio={rec['ratio']:.3f} "
                  f"isi={rec['fill']*100:4.0f}%", flush=True)
        n = end
        blk += 1

    fh.close()
    print(f"ditulis -> {a.jsonl}", flush=True)


if __name__ == "__main__":
    main()
