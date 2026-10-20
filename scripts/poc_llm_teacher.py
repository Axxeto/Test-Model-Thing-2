"""PoC — APAKAH "TEACHER" EKSTERNAL (mis. LLM) BERGUNA untuk TMT2?

Fakta arsitektur (terverifikasi): `stat_step` SUDAH punya masukan eksternal —
`st[nctx+1] = nn_row[bi]` = stretch (log-odds natural) p(bit|prefiks) untuk 8 bit byte ini,
dengan bobot mixer TERPELAJAR (`w1[:, :, nctx+1]`). Mixer = jumlah terbobot di ruang logit,
jadi ini persis ansambel LOG-LINEAR dua model probabilitas. LLM bisa dicangkok tanpa ubah
arsitektur.

Yang diukur di sini: kurva **kualitas guru -> kualitas gabungan**. Guru disimulasikan
sebagai oracle + derau pada logit:  t_teacher = t_benar + N(0, sigma).
  sigma=0   -> guru sempurna (BPB~0)  [sanity]
  sigma besar -> guru makin tak berguna (BPB guru naik)
Dilaporkan: BPB guru SENDIRI, BPB TMT2+guru, dan BPB TMT2 tanpa guru.

Kontrol WAJIB (aturan proyek #7): mode "random".

Jalan:  python scripts/poc_llm_teacher.py --corpus wikitext --ntrain 200000 --ntest 8000
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.stat_core import StatCore

L_BIG = 12.0            # logit "benar" (p = sigmoid(12) ~ 0,999994)


def _true_logit(data):
    """(N,8) logit oracle: +L_BIG bila bit=1, -L_BIG bila bit=0 (MSB-first)."""
    d = np.asarray(data)
    out = np.empty((len(d), 8), dtype=np.float64)
    for bi in range(8):
        bit = ((d >> (7 - bi)) & 1).astype(np.float64)
        out[:, bi] = np.where(bit > 0.5, L_BIG, -L_BIG)
    return out


def make_teacher(data, mode, sigma, rng):
    """Kembalikan (nn_stream | None, bpb_guru | None)."""
    if mode == "none":
        return None, None
    if mode == "random":
        return rng.normal(0.0, 2.0, size=(len(data), 8)), None
    t = _true_logit(data)
    if sigma > 0:
        t = t + rng.normal(0.0, sigma, size=t.shape)
    # BPB guru SENDIRI: -log2 p(bit benar) dijumlah 8 bit
    d = np.asarray(data)
    ce = 0.0
    for bi in range(8):
        bit = ((d >> (7 - bi)) & 1).astype(np.float64)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(t[:, bi], -30, 30)))
        p = np.where(bit > 0.5, p1, 1.0 - p1)
        ce += float(np.mean(-np.log2(np.clip(p, 1e-12, 1.0))))
    return t, ce


def run(corpus, ntrain, ntest, mode, sigma, tb, orders, seed=0):
    raw = np.frombuffer(load(corpus), dtype=np.uint8)
    tr, te = raw[:ntrain], raw[ntrain:ntrain + ntest]
    rng = np.random.default_rng(seed)
    ttr, _ = make_teacher(tr, mode, sigma, rng)
    tte, guru_bpb = make_teacher(te, mode, sigma, rng)

    m = StatCore(orders=orders, table_bits=tb, wsel_bits=8, match_bits=16,
                 match_fix=True, ctx_fix=True, use_apm=True, mixer_mode=1)
    t0 = time.perf_counter()
    for i, b in enumerate(tr):
        m.step(int(b), train=True, nn_st=(None if ttr is None else ttr[i]))
    el = time.perf_counter() - t0
    bpb = float(np.mean(m.eval_stream(te, nn_stream=tte)))
    return dict(mode=mode, sigma=sigma, guru_bpb=guru_bpb, bpb=bpb,
                w_teacher=float(np.mean(np.abs(m.w1[:, :, m.nctx + 1]))),
                bps=round(len(tr) / max(el, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200_000)
    ap.add_argument("--ntest", type=int, default=8_000)
    ap.add_argument("--table_bits", type=int, default=20)
    ap.add_argument("--orders", default="0,1,2,3,4,5,6,8")
    ap.add_argument("--sigmas", default="0,0.5,1,2,4,8")
    ap.add_argument("--jsonl", default="results/poc_llm_teacher.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    sigmas = [float(x) for x in a.sigmas.split(",")]
    print(f"[PoC] korpus={a.corpus} latih={a.ntrain} uji={a.ntest} tb={a.table_bits}", flush=True)
    rows = [run(a.corpus, a.ntrain, a.ntest, "none", 0.0, a.table_bits, orders),
            run(a.corpus, a.ntrain, a.ntest, "random", 0.0, a.table_bits, orders)]
    for s in sigmas:
        rows.append(run(a.corpus, a.ntrain, a.ntest, "noisy", s, a.table_bits, orders))

    base = next(r for r in rows if r["mode"] == "none")
    print(f"\n  {'mode':10s} {'BPB_guru':>9s} {'BPB_TMT2+guru':>14s} {'Δ vs tanpa guru':>16s} "
          f"{'w_guru':>7s}")
    for r in rows:
        g = "   —   " if r["guru_bpb"] is None else f"{r['guru_bpb']:9.4f}"
        print(f"  {r['mode']:10s} {g} {r['bpb']:14.4f} {r['bpb']-base['bpb']:+16.4f} "
              f"{r['w_teacher']:7.3f}")
    print(f"\n  (tanpa guru = {base['bpb']:.4f} bit/byte; kontrol guru-acak harus ~0)")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, ntest=a.ntest,
                                table_bits=a.table_bits, rows=rows)) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
