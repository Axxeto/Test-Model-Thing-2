"""G-small/G-mid: matriks arm V0/V2/V3/V3S × korpus × offset data (T0.5).

Replikasi = OFFSET DATA (StatCore deterministik; "seed" tak bermakna).
Uji = 5 jendela × win tepat setelah latih; Δ BERPASANGAN vs V0 per jendela + SE.

Contoh:
  python scripts/g1_v3_matrix.py --corpora wikitext,indonesian --offsets 0,250000,500000 \
      --ntrain 200000 --ntest 40000 --win 8000 --arms V0,V2,V3,V3S
  python scripts/g1_v3_matrix.py --corpora big --offsets 0 --ntrain 1000000 \
      --ntest 40000 --win 8000 --arms V0,V2,V3,V3S
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.corpus import load
from tmt2.stat_core import StatCore

ensure_utf8()
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)

ARMS = {
    "V0": dict(mixer_mode=0),
    "V2": dict(mixer_mode=1),
    "V3": dict(mixer_mode=1, mix_lr_nhalf=50000, mix_lr_floor=0.02),
    "V3S": dict(mixer_mode=1, mix_lr_nhalf=200000, mix_lr_floor=0.02),
    # Tanpa-kernel (C1): selector kelas-karakter, V3H (lr awal 2x + floor 0,5),
    # dan match_bits 24. Kunci `_lr`/`_wsel`/`_mb` diproses di bawah.
    "WSEL": dict(mixer_mode=1, _wsel=1),
    "V3H": dict(mixer_mode=1, _lr=0.004, mix_lr_floor=0.5,
                mix_lr_nhalf=50000),
    "MB24": dict(mixer_mode=1, _mb=24),
    # Adopsi: dropout-latih + C2-L1 (kernel).
    "DROP": dict(mixer_mode=1, dropout_permille=100),
    "C2": dict(mixer_mode=1, match_adapt=1),
    # A2: selector set-0 per hit-order (opt[1], kernel).
    "HITSEL": dict(mixer_mode=1, hit_sel=1),
}


def get_corpus(name):
    if name == "wikitext":
        return np.frombuffer(load("wikitext"), dtype=np.uint8)
    if name == "indonesian":
        return np.frombuffer(load("data/indonesian_1m_pure.txt"), dtype=np.uint8)
    if name == "big":
        with open("data/indonesian_big.bin", "rb") as f:
            return np.frombuffer(f.read(200_000_000), dtype=np.uint8)
    raise SystemExit(f"korpus tak dikenal: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="wikitext,indonesian")
    ap.add_argument("--offsets", default="0,250000,500000")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=40000)
    ap.add_argument("--win", type=int, default=8000)
    ap.add_argument("--arms", default="V0,V2,V3,V3S")
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--jsonl", default="results/g1_v3_matrix.jsonl")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    nwin = a.ntest // a.win

    with open(a.jsonl, "a", encoding="utf-8") as f:
        for corpus in [c.strip() for c in a.corpora.split(",") if c.strip()]:
            raw = get_corpus(corpus)
            for off in [int(x) for x in a.offsets.split(",") if x.strip() != ""]:
                tr = raw[off:off + a.ntrain].copy()
                wins = [raw[off + a.ntrain + i * a.win:
                            off + a.ntrain + (i + 1) * a.win].copy()
                        for i in range(nwin)]
                per = {}
                for arm in arms:
                    kw = dict(ARMS[arm])
                    lr = float(kw.pop("_lr", 0.002))
                    wsel = int(kw.pop("_wsel", 0))
                    mb = int(kw.pop("_mb", 20))
                    m = StatCore(orders=ORDERS, table_bits=a.table_bits, lr=lr,
                                 use_apm=True, apm_lr=0.02, match_minlen=6,
                                 match_bits=mb, match_conf=2.0, wsel_bits=12,
                                 sparse=False, word=True, count_cap=1023,
                                 cks_bits=24, match_fix=True, ctx_fix=True,
                                 wsel_mode=wsel, **kw)
                    m.train_stream(tr)
                    bpb, pw = m.eval_suite(wins, cold=True)
                    per[arm] = pw
                    print(f"[{corpus} off={off} {arm}] BPB {bpb:.4f}", flush=True)
                base = np.asarray(per[arms[0]])
                row = dict(corpus=corpus, off=off, ntrain=a.ntrain, arms=arms,
                           baseline=arms[0])
                for arm in arms[1:]:
                    d = np.asarray(per[arm]) - base
                    row[arm] = dict(delta=round(float(d.mean()), 5),
                                    se=round(float(d.std() / len(d) ** 0.5), 5))
                print(f"  Δvs{arms[0]}: " + " ".join(
                    f"{arm} {row[arm]['delta']:+.5f}±{row[arm]['se']:.5f}"
                    for arm in arms[1:]), flush=True)
                f.write(json.dumps(row) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
