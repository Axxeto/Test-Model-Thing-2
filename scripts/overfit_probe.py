"""GERBANG OVERFIT: bisakah model menghafal N byte?

Ukur CE BEKU (state dingin, tanpa belajar) setelah N epoch -- bukan hanya CE latih.
Ini uji paling murah yang memisahkan dua diagnosis yang saling eksklusif:
  - BISA menghafal -> kapasitas + optimasi memadai; plateau = rezim data / keying.
  - TIDAK bisa     -> optimasi atau gradien yang menghambat -> ChunkLearner jadi prioritas.

Grid kecil --layers x --deep menjawab U4 (apakah tumpukan 3 lapis dipakai) SEKALIGUS
pertanyaan gradien lintas lapisan, tanpa run tambahan.

Contoh:
  python scripts/overfit_probe.py --n 1000 --epochs 200 --layers 1,3 --deep 0,1 --lrs 1e-2 --clips 1
  python scripts/overfit_probe.py --n 512  --epochs 300 --data periodic
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load

LN2 = float(np.log(2))


def min_unique_k(data, kmax=64):
    """k terkecil sehingga semua k-gram unik (semua posisi dapat diidentifikasi dari k byte).

    Bila k kecil (<=8) maka kegagalan overfit BUKAN soal resolusi konteks.
    Mengembalikan (k, n_gram) atau (None, None) bila kmax terlampaui.
    """
    n = len(data)
    for k in range(1, kmax + 1):
        grams = {bytes(data[i:i + k]) for i in range(n - k + 1)}
        if len(grams) == n - k + 1:
            return k, n - k + 1
    return None, None


def run(data, d, layers, deep, lr, se, clip, epochs, seed, d_mem=None):
    cfg = Config(d=d, n_layers=layers, d_mem=(d_mem if d_mem else max(8, d // 4)),
                 head="linear", deep_grad=bool(deep), lr=lr, warmup=200, slow_every=se,
                 clip=clip, wd=0.0, replay=False, adapter_rank=0, use_innovation=False,
                 mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)
    curve, t0 = [], time.time()
    for _ in range(epochs):
        ce = [c for c in (ln.step(s, int(b)) for b in data) if c is not None]
        curve.append(float(np.mean(ce) / LN2) if ce else float("nan"))
    frozen = float(ln.eval_stream([int(b) for b in data]).mean() / LN2)  # state dingin, tanpa belajar
    first = lambda thr: next((i + 1 for i, v in enumerate(curve) if v < thr), None)
    return dict(frozen=frozen, ep1=first(1.0), ep01=first(0.1), ep1_5=first(1.5),
                adam=ln.t_adam, secs=time.time() - t0, last_train=curve[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="jumlah byte yang dihafal")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="license")
    ap.add_argument("--d_mem", type=int, default=None)
    ap.add_argument("--layers", default="1,3", help="daftar n_layers, mis. '1,3'")
    ap.add_argument("--deep", default="0,1", help="daftar deep_grad, mis. '0,1'")
    ap.add_argument("--lrs", default="1e-2,3e-2", help="daftar lr")
    ap.add_argument("--ses", default="1,4", help="daftar slow_every")
    ap.add_argument("--clips", default="1,10", help="daftar clip")
    a = ap.parse_args()

    raw = load(a.data)
    data = raw[:a.n]
    k, ngram = min_unique_k(data)
    ref = 8.0 if a.data != "periodic" else 8.0   # log2(256)
    print(f"data={a.data}  n={a.n}  byte unik={len(set(data))}  "
          f"k-gram-unik pertama k={k} (n_gram={ngram})")
    print(f"lantai seragam = {np.log2(256):.2f} bit  |  target beku < 0.1 bit  |  "
          f"epochs={a.epochs} d={a.d} seed={a.seed} adapter_rank=0 innovation=off")
    hdr = (f"{'L':>2}{'deep':>5}{'lr':>8}{'se':>4}{'clip':>6}{'beku':>9}"
           f"{'latih_akhir':>12}{'ep<1.5b':>9}{'ep<1b':>8}{'ep<.1b':>8}{'Adam':>7}{'detik':>7}")
    print(hdr)
    print("-" * len(hdr))
    best = None
    for L, dp, lr, se, cl in itertools.product(
            [int(x) for x in a.layers.split(",")],
            [int(x) for x in a.deep.split(",")],
            [float(x) for x in a.lrs.split(",")],
            [int(x) for x in a.ses.split(",")],
            [float(x) for x in a.clips.split(",")]):
        r = run(data, a.d, L, dp, lr, se, cl, a.epochs, a.seed, a.d_mem)
        if best is None or r["frozen"] < best["frozen"]:
            best = dict(r, L=L, deep=dp, lr=lr, se=se, clip=cl)
        fmt = lambda v: "-" if v is None else str(v)
        print(f"{L:>2}{dp:>5}{lr:>8.0e}{se:>4}{cl:>6.0f}{r['frozen']:>9.3f}"
              f"{r['last_train']:>12.3f}{fmt(r['ep1_5']):>9}{fmt(r['ep1']):>8}"
              f"{fmt(r['ep01']):>8}{r['adam']:>7}{r['secs']:>7.0f}", flush=True)
    print("-" * len(hdr))
    if best:
        print(f"TERBAIK: L={best['L']} deep={best['deep']} lr={best['lr']:.0e} "
              f"se={best['se']} clip={best['clip']:.0f}  beku={best['frozen']:.3f} bit")

    # -- bacaan otomatis
    print("\n== bacaan ==")
    if best and best["frozen"] < 0.1:
        print("BISA menghafal (<0.1 bit) -> kapasitas + optimasi memadai.")
        print("  => prioritaskan kurva data & kepala (Fase 3), bukan ChunkLearner.")
    else:
        print("TIDAK menghafal di grid ini -> hambatan optimasi/gradien NYATA.")
        print("  => ChunkLearner (gradien lintas lapisan benar) jadi prioritas.")
    if k is not None and k <= 8:
        print(f"  k-gram-unik k={k} kecil -> kegagalan BUKAN soal resolusi konteks.")


if __name__ == "__main__":
    main()
