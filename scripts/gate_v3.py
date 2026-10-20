"""Gerbang §7 Mixer v3: V0 vs V3 pada 30MB, tb25/26, urutan asli/acak, suite TETAP.

Perbedaan vs E-series: jendela uji TAK BERTUMPUK (chunk sekuensial, bukan acak
dalam 4MB — kritik §7 soal SE optimistis), daerah uji mulai TEPAT di byte
ntrain sehingga run ntrain sama dapat dibandingkan, dan kurva titik (default
3/6/9/12/18/30jt) diukur pada suite yang SAMA.

Urutan acak = block-shuffle (default 100kB) pada daerah LATIH saja dgn seed
tetap; daerah uji identik antar urutan.

Contoh (~8 mnt/run):
  python scripts/gate_v3.py --table_bits 25 --variant V3 --jsonl results/gate_v3.jsonl
  python scripts/gate_v3.py --table_bits 26 --variant V0 --order acak
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()

ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)
CURVE = (3_000_000, 6_000_000, 9_000_000, 12_000_000, 18_000_000, 30_000_000,
         60_000_000, 100_000_000, 150_000_000, 186_000_000)

ARMS = ("V0", "V2", "V3", "V3S")


def build(a, variant):
    kw = dict(orders=ORDERS, table_bits=a.table_bits, lr=0.002,
              use_apm=True, apm_lr=0.02, match_minlen=6, match_bits=a.match_bits,
              match_conf=2.0, wsel_bits=12, sparse=a.sparse, word=True,
              count_cap=1023, cks_bits=24, match_fix=True, ctx_fix=True,
              assoc=a.assoc, dropout_permille=a.drop, hit_sel=a.hitsel)
    if variant == "V2":
        kw.update(mixer_mode=1)
    elif variant == "V3":
        kw.update(mixer_mode=1, mix_lr_nhalf=50000, mix_lr_floor=0.02)
    elif variant == "V3S":
        kw.update(mixer_mode=1, mix_lr_nhalf=200000, mix_lr_floor=0.02)
    return StatCore(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--variant", default="V3", choices=ARMS)
    ap.add_argument("--order", default="asli", choices=("asli", "acak"))
    ap.add_argument("--block", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sparse", action="store_true",
                    help="nyalakan konteks jarang (default MATI; E-series memakai ON)")
    ap.add_argument("--assoc", type=int, default=0,
                    help="1 = bucket 4-way + eviksi hitungan (#4, arm B)")
    ap.add_argument("--match_bits", type=int, default=20,
                    help="bit tabel match (20 default; 24 = horizon ~16x, A4)")
    ap.add_argument("--drop", type=int, default=0,
                    help="dropout konteks permille saat latih (0=mati; 100=sepuluh persen)")
    ap.add_argument("--hitsel", type=int, default=0,
                    help="1 = selector set-0 per hit-order (A2)")
    ap.add_argument("--nwin", type=int, default=20)
    ap.add_argument("--win", type=int, default=200_000)
    ap.add_argument("--suite_at", type=int, default=-1,
                    help="awal byte daerah uji (default=-1: tepat setelah ntrain)")
    ap.add_argument("--jsonl", default="results/gate_v3.jsonl")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()

    suite_at = a.ntrain if a.suite_at < 0 else a.suite_at
    need = max(a.ntrain, suite_at) + a.nwin * a.win
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    if len(raw) < need:
        raise SystemExit(f"korpus kurang: {len(raw)} < {need}")
    tr = raw[:a.ntrain].copy()
    if a.order == "acak":
        rng = np.random.default_rng(a.seed)
        nb = len(tr) // a.block
        perm = rng.permutation(nb)
        tr = np.concatenate([tr[i * a.block:(i + 1) * a.block] for i in perm])
        print(f"[gerbang] urutan ACAK blok-{a.block} seed {a.seed}", flush=True)
    # jendela tak-bertumpuk, sekuensial dari awal daerah uji (bisa beda dari ntrain)
    test_region = raw[suite_at:suite_at + a.nwin * a.win]
    windows = [test_region[i * a.win:(i + 1) * a.win].copy() for i in range(a.nwin)]

    m = build(a, a.variant)
    print(f"[gerbang] {a.variant} tb={a.table_bits} {a.order} latih {a.ntrain}B "
          f"suite@{suite_at/1e6:.0f}M", flush=True)
    t0 = time.perf_counter()
    curve, per_win, prev = {}, {}, 0
    for nb in sorted(set(c for c in CURVE if c <= a.ntrain) | {a.ntrain}):
        if nb > prev:
            m.train_stream(tr[prev:nb])
            prev = nb
        bpb, per = m.eval_suite(windows, cold=True)
        curve[nb] = round(bpb, 4)
        per_win[nb] = [round(x, 5) for x in per]  # WAJIB: SE berpasangan
        el = time.perf_counter() - t0
        print(f"  {nb/1e6:5.1f}jt: BPB {bpb:.4f} ({el:.0f}s, {prev/max(el,1e-9):.0f} B/s)",
              flush=True)
    d = m.diagnose(windows[0][:20000])
    fill = float(np.count_nonzero(m.tab) / m.tab.size)
    rec = dict(variant=a.variant, table_bits=a.table_bits, order=a.order,
               sparse=bool(a.sparse), assoc=int(a.assoc), drop=int(a.drop),
               hitsel=int(a.hitsel), match_bits=int(a.match_bits),
               ntrain=a.ntrain, suite_at=suite_at,
               curve=curve, per_win=per_win,
               w1max=round(d["w1max"], 1),
               sat=[round(s, 3) for s in d["sat_frac"]],
               ownCE=[round(v * 8, 2) for v in d["own_ce"]],
               w2neg=round(d["w2neg_frac"], 4), fill=round(fill, 4))
    print(f"[gerbang] {a.variant}/{a.order}/tb{a.table_bits}: {curve} "
          f"max|w1|={rec['w1max']} sat={rec['sat']}", flush=True)
    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if a.ckpt:
        m.save(a.ckpt, dict(gate=rec))
        print(f"ckpt -> {a.ckpt}", flush=True)


if __name__ == "__main__":
    main()
