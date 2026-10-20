"""UJI HAFALAN MURNI: 512 byte ACAK V=64 (pasti bisa dihafal oleh model ~300k parameter).

Tujuan: memisahkan "model tidak bisa fitting" dari "teks alami butuh konteks panjang".
Pada byte acak V=64 (entropi 6 bit/byte), tidak ada struktur untuk dipelajari -- satu-satunya
cara menurunkan CE beku adalah MENGHAPAL. Karena itu, bila model tidak bisa <0.1 bit di sini
maka ada hambatan optimasi/gradien yang nyata, terlepas dari teks.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import random_bytes

LN2 = float(np.log(2))


def run(data, L, dp, se, epochs=80, d=32, lr=3e-2, d_mem=8):
    cfg = Config(d=d, n_layers=L, d_mem=d_mem, head="linear", deep_grad=bool(dp), lr=lr,
                 warmup=200, slow_every=se, clip=1.0, wd=0.0, replay=False, adapter_rank=0,
                 use_innovation=False, mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=0)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m); t0 = time.time(); curve = []
    for _ in range(epochs):
        ce = [c for c in (ln.step(s, int(b)) for b in data) if c is not None]
        curve.append(float(np.mean(ce) / LN2))
    fz = float(ln.eval_stream([int(b) for b in data]).mean() / LN2)
    f = lambda t: next((i + 1 for i, v in enumerate(curve) if v < t), None)
    return fz, curve[-1], f(1.0), f(0.1), time.time() - t0, curve


def main():
    data = random_bytes(512, 64, 7)
    print("unit: 512 byte ACAK V=64  (entropi=6 bit/byte; bisa dihafal -> beku mendekati 0)", flush=True)
    print(f"{'L':>2}{'deep':>5}{'se':>4}{'beku':>9}{'latih_akhir':>12}{'ep<1b':>7}{'ep<.1b':>8}{'detik':>7}")
    print("-" * 56)
    for L, dp, se in [(1, 1, 4), (1, 1, 16), (3, 1, 4), (3, 1, 16), (3, 0, 4), (3, 0, 16)]:
        fz, last, e1, e01, dt, _ = run(data, L, dp, se)
        fmt = lambda v: "-" if v is None else str(v)
        print(f"{L:>2}{dp:>5}{se:>4}{fz:>9.3f}{last:>12.3f}{fmt(e1):>7}{fmt(e01):>8}{dt:>7.0f}", flush=True)


if __name__ == "__main__":
    main()
