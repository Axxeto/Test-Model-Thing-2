"""Instrumentasi langsung: KOMPONEN MANA yang memburuk seiring latihan panjang?

Diuji & DIBUANG secara empiris (jangan ulangi):
  * ukuran tabel (`table_bits` 25/26/27) — semua konvergen ~2,28 @30jt.
  * `count_cap` (1023 vs 4095) — BPB 2,2745 vs 2,2744 (NOL).
  * lebar checksum (`cks_bits` 8/16/24) — 2,470495/2,470172/2,470173 (−0,0003).
Ketiganya BUKAN penyebabnya. Skrip ini mengukur komponen secara langsung dengan
membekukan bagian-bagian dan melihat mana yang menanggung degradasi.

Cara ukur: pada tiga titik (10/20/30 jt byte) simpan snapshot penuh state (tabel,
penghitung mixer, APM), lalu di SETIAP titik evaluasi jendela uji yang sama dengan
beberapa VARIAN pembekuan:
  A. penuh            : apa adanya
  B. APM dibekukan    : use_apm=0 saat eval (uji apakah APM yang melenceng)
  C. mixer dibekukan  : w1/w2 di-reset ke seragam (uji apakah mixer yang melenceng)
  D. tabel di-reset   : tab dinolkan (uji apakah tabel menyimpan statistik buruk)
Perbedaan antar-variant menunjukkan komponen mana yang menyumbang degradasi.

Jalan:
  python scripts/diag_component.py --ntrain 30000000 --ntest 20000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore, build_apm

CORPUS = "data/indonesian_big.bin"


def snapshot(m):
    return dict(tab=m.tab.copy(), w1=m.w1.copy(), w2=m.w2.copy(),
                apm1=m.apm1.copy(), apm2=m.apm2.copy())


def restore(m, s):
    m.tab[:] = s["tab"]
    m.w1[:] = s["w1"]
    m.w2[:] = s["w2"]
    m.apm1[:] = s["apm1"]
    m.apm2[:] = s["apm2"]


def variant_bpb(m, te, variant):
    """BPB jendela uji untuk satu varian pembekuan, lalu pulihkan state."""
    s = snapshot(m)
    try:
        if variant == "penuh":
            pass
        elif variant == "tanpa_apm":
            m.use_apm = False
        elif variant == "mixer_seragam":
            m.w1[:] = 1.0 / m.nctx
            m.w2[:] = 0.0
            m.w2[:, :3] = 1.0 / 3.0
        elif variant == "tabel_kosong":
            m.tab[:] = 0
        else:
            raise ValueError(variant)
        ce = float(m.eval_stream(te).mean())
    finally:
        m.use_apm = True
        restore(m, s)
    return ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30000000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--match_fix", type=int, default=1)
    ap.add_argument("--ctx_fix", type=int, default=1)
    ap.add_argument("--probe_every", type=int, default=10000000)
    ap.add_argument("--jsonl", default="results/diag_component.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    tr = raw[:a.ntrain]
    # ⚠️ Jendela uji: dokumen ACAK tersebar (bukan irisan 20kb yang bising).
    # Ambil ~1% dokumen dari 0..ntrain, buang dari latih. (Perbaikan metodologi.)
    data = bytes(raw[:a.ntrain])
    docs = [d for d in data.split(b"\n\n") if len(d) > 20]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(docs))
    n_test = max(1, len(docs) // 100)
    test_set = set(perm[:n_test].tolist())
    te = np.frombuffer(b"\n\n".join(docs[i] for i in sorted(test_set)), dtype=np.uint8)
    tr = np.frombuffer(b"\n\n".join(docs[i] for i in perm if i not in test_set),
                       dtype=np.uint8)
    print(f"latih {len(tr):,} B | uji(Acak tersebar) {len(te):,} B ({n_test} dok) | "
          f"tb={a.table_bits} cks={a.cks_bits} match_fix={a.match_fix} ctx_fix={a.ctx_fix}",
          flush=True)
    print("varian: penuh / tanpa_apm / mixer_seragam / tabel_kosong\n", flush=True)

    m = StatCore(orders=orders, table_bits=a.table_bits, lr=a.lr, use_apm=True,
                 wsel_bits=12, sparse=False, word=True, cks_bits=a.cks_bits,
                 match_fix=bool(a.match_fix), ctx_fix=bool(a.ctx_fix))
    variants = ["penuh", "tanpa_apm", "mixer_seragam", "tabel_kosong"]
    os.makedirs("results", exist_ok=True)
    rows = []

    n = 0
    t0 = time.perf_counter()
    next_probe = a.probe_every
    hdr = f"{'byte':>11} " + " ".join(f"{v:>14}" for v in variants)
    print(hdr, flush=True)
    for b in tr:
        m.step(b)
        n += 1
        if n >= next_probe:
            vals = {}
            for v in variants:
                vals[v] = variant_bpb(m, te, v)
            print(f"{n:>11,} " + " ".join(f"{vals[v]:>14.4f}" for v in variants)
                  + f"   ({(n/max(time.perf_counter()-t0,1e-9)):.0f} B/s)", flush=True)
            rows.append(dict(n=n, table_bits=a.table_bits, **vals))
            next_probe += a.probe_every

    with open(a.jsonl, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
