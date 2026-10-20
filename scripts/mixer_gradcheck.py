"""Gradcheck mixer konteks (Pilar 2) — versi bersih.

Strategi: MATIKAN optimizer (slow_every sangat besar + lr=0) supaya `gacc` TIDAK pernah
dibersihkan, lalu baca gradien akumulatif untuk task "byte". Bandingkan dgn beda-hingga
terpusat pada `mix_w` saja (tabel n-gram dibangun identik deterministik).

PENTING (pelajaran proyek):
  * Jangan baca `gacc` setelah lintasan penuh tanpa memastikan `slow_update` tidak pernah jalan
    -- `slow_update` membersihkan `gacc` (fast_learner.py:856).
  * Jangan `m.mix_w.data = w` (memutus tampilan arena) -> pakai `m.mix_w.data[:] = w`.
  * Set bobot SEBELUM fase konteks, jika tidak ada suku CE "warisan" di entri batas t-1.
  * Sumber Markov di sini memakai generator DETERMINISTIK dari `tmt2.corpus.markov_det_stream`
    (lantai eps*log2 V), bukan generator kolom-acak (yang lantainya ~2,25 bit).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner
from tmt2.corpus import markov_det_stream


def build_cfg():
    return Config(vocab=8, d=8, n_layers=1, head="linear", use_memory=False,
                  mixer_k=1, mixer_match=True, mixer_match_orders="1,2,3",
                  mixer_init=0.05, ngram_learn=False, deep_grad=False, s_decay=0.0,
                  slow_every=10 ** 9, lr=0.0, fast_lr=0.0)


def opts_frozen(seed=1):
    # slow_every besar (di Config) -> tak pernah update; lr=0 -> identik walau update jalan
    return TrainOpts(seed=seed, think_prob=0.0, only_byte_loss=True, use_stop=False,
                     ngram_lam=0.0)


def _run(data_ctx, data_grad, wvec):
    """Satu lintasan: set bobot SEBELUM fase ctx (sehingga SEMUA entri memakai wvec yang sama,
    menghindari suku CE 'warisan' dari entri batas yang tidak ikut terganggu). Kembalikan
    (sum CE atas langkah grad, gradien mix_w akumulatif)."""
    m = TMT2(build_cfg())
    fl = FastLearner(m, opts_frozen())
    m.mix_w.data[:] = wvec                            # IN-PLACE, sebelum ctx
    s = Stream(m, online=True)
    for b in data_ctx:
        fl.step(s, int(b), train=True, think=False, fast_ok=True)
    fl.gacc = {}
    tot = 0.0
    for b in data_grad:
        ce = fl.step(s, int(b), train=True, think=False, fast_ok=True)
        if ce is not None:
            tot += float(ce)
    g = fl.gacc.get("byte")
    off, shp = fl.offs["mix_w"]
    ga = (g[off:off + int(np.prod(shp))].copy() if g is not None else np.zeros_like(wvec))
    return tot, ga


def main():
    data, floor = markov_det_stream(500, 2, 8, eps=0.15, seed=7)
    print(f"sumber: markov deterministik k=2 V=8 eps=0.15 (lantai {floor:.3f} bit)")
    ctx, grad = data[:200], data[200:450]

    w0 = np.full(4, 0.05)
    _, g_an = _run(ctx, grad, w0)
    eps = 1e-6
    g_fd = np.zeros_like(w0)
    for j in range(len(w0)):
        wp = w0.copy(); wp[j] += eps
        wm = w0.copy(); wm[j] -= eps
        a, _ = _run(ctx, grad, wp)
        b, _ = _run(ctx, grad, wm)
        g_fd[j] = (a - b) / (2 * eps)

    rel = np.abs(g_an - g_fd) / (np.abs(g_fd) + 1e-9)
    print("g_an :", np.array2string(g_an, precision=7))
    print("g_fd :", np.array2string(g_fd, precision=7))
    print("rel  :", np.array2string(rel, precision=3))
    print("maks rel err =", float(rel.max()))
    ok = rel.max() < 1e-5
    print("HASIL:", "LULUS" if ok else "GAGAL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
