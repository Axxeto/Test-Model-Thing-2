"""BENCH Tingkat A (inti statistik biner) — protokol G1 dari ARCHITECTURE_V2.md §7.

Latih pada data[:ntrain], nilai pada data[ntrain:ntrain+ntest] yang TAK PERNAH dilihat.
Melaporkan BPB tahan + throughput B/s STEADI-STATE (JIT dipanaskan lebih dulu).

Pembanding (sudah terukur, protokol sama):
  TMT2+mixer (simbol) WikiText 1,8264 ± 0,0014 (3 seed) · Indonesia 2,4809 ± 0,0068
  n-gram JM-5 beku     WikiText 1,9254 · Indonesia 2,5400
  bz2 -9 kondisional   WikiText 1,9450 · Indonesia 2,3700
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.corpus import load
from tmt2.stat_core import StatCore

ensure_utf8()

CORPORA = {"wikitext": "wikitext", "indonesian": "data/indonesian_1m_pure.txt"}


def run(kind, ntrain, ntest, table_bits, lr, orders, warm=3000, use_apm=True, apm_lr=0.02,
        match_minlen=6, match_bits=20, match_conf=2.0, wsel_bits=8, sparse=False, word=True):
    raw = np.frombuffer(load(CORPORA[kind]), dtype=np.uint8)
    tr = raw[:ntrain]
    te = raw[ntrain:ntrain + ntest]
    m = StatCore(orders=orders, table_bits=table_bits, lr=lr, use_apm=use_apm, apm_lr=apm_lr,
                 match_minlen=match_minlen, match_bits=match_bits, match_conf=match_conf,
                 wsel_bits=wsel_bits, sparse=sparse, word=word)
    # pemanasan: JIT + tabel terisi sedikit
    for b in tr[:warm]:
        m.step(b)
    # ukur throughput steadi-state pada sisa data latih
    rest = tr[warm:]
    t0 = time.perf_counter()
    for b in rest:
        m.step(b)
    dt = time.perf_counter() - t0
    bps = len(rest) / dt
    ce = m.eval_stream(te)
    return float(ce.mean()), bps, len(te), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--table_bits", type=int, default=22)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--corpora", default="wikitext,indonesian")
    ap.add_argument("--skip_train", action="store_true",
                    help="hanya throughput (tak perlu BPB tahan)")
    ap.add_argument("--no_apm", action="store_true", help="matikan APM/SSE (ablasi)")
    ap.add_argument("--apm_lr", type=float, default=0.02)
    ap.add_argument("--match_minlen", type=int, default=6)
    ap.add_argument("--match_bits", type=int, default=20)
    ap.add_argument("--match_conf", type=float, default=2.0)
    ap.add_argument("--no_match", action="store_true", help="matikan model match (ablasi)")
    ap.add_argument("--no_sparse", action="store_true", default=True,
                    help="konteks jarang MERUGI (+0,038 bit) → default MATI (produksi)")
    ap.add_argument("--sparse", dest="no_sparse", action="store_false",
                    help="nyalakan konteks jarang (ablasi)")
    ap.add_argument("--no_word", action="store_true", help="matikan konteks kata (ablasi)")
    ap.add_argument("--wsel_bits", type=int, default=8,
                    help="bit pemilih set bobot mixer (8 = hanya c0)")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    print(f"TINGKAT A · ntrain={a.ntrain} ntest={a.ntest} table_bits={a.table_bits} "
          f"lr={a.lr} orders={orders}")
    print(f"memori tabel ≈ {3 * (1 << a.table_bits) / 1e6:.1f} MB "
          f"(chk 1B + n0 2B + n1 2B per slot)\n")
    hdr = f"{'korpus':<12}{'BPB tahan':>12}{'B/s':>12}{'n uji':>8}"
    print(hdr)
    print("-" * len(hdr))
    for kind in [c.strip() for c in a.corpora.split(",") if c.strip()]:
        bpb, bps, n, m = run(kind, a.ntrain, a.ntest, a.table_bits, a.lr, orders,
                             use_apm=not a.no_apm, apm_lr=a.apm_lr,
                             match_minlen=(10 ** 9 if a.no_match else a.match_minlen),
                             match_bits=a.match_bits, match_conf=a.match_conf,
                             wsel_bits=a.wsel_bits, sparse=not a.no_sparse,
                             word=not a.no_word)
        print(f"{kind:<12}{bpb:>12.4f}{bps:>12.0f}{n:>8}", flush=True)

    print("\n== pembanding (protokol sama) ==")
    print("  TMT2+mixer (simbol, 3 seed)  WikiText 1,8264 · Indonesia 2,4809")
    print("  n-gram JM-5 beku             WikiText 1,9254 · Indonesia 2,5400")
    print("  bz2 -9 kondisional           WikiText 1,9450 · Indonesia 2,3700")
    print("\nG1: Tingkat A harus menang ≥0,05 bit atas mixer simbol (WikiText) DAN")
    print("    menutup gap ke bz2 (Indonesia), serta ≥100k B/s pada 1 inti.")


if __name__ == "__main__":
    main()
