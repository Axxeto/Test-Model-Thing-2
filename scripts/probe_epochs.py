"""PROBE EPOCH — apakah hambatan kualitas = REZIM ONLINE (satu lintasan), bukan arsitektur?

Latih pada data[:N] selama E epoch (urutan sama, ONLINE di dalam epoch), ukur CE TAHAN
(tanpa belajar, state dingin) setelah tiap epoch.

Bila CE tahan turun JAUH dari epoch 1 ke epoch berikutnya -> yang menghambat adalah
optimasi/anggaran lintasan, bukan kapasitas arsitektur. Bila datar -> arsitektur/rezim
data yang membatasi (dan menambah epoch hanya menghafal).

Kontrol: CE LATIH juga dilaporkan, supaya bisa dibedakan "belajar lebih baik" vs "menghafal".
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=100000)
    ap.add_argument("--nheld", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixer", type=int, default=0, help=">0: aktifkan mixer orde 1..N (tanpa fused)")
    ap.add_argument("--zero_out", type=int, default=0, help="1: W_out=0 (mixer murni, SSM tak masuk logits)")
    ap.add_argument("--mixer_init", type=float, default=0.0)
    ap.add_argument("--mixer_mult", type=float, default=0.1)
    ap.add_argument("--match", type=int, default=0, help="1: aktifkan match model mixer")
    ap.add_argument("--match_recent", type=int, default=0,
                    help="1: match 'kemunculan terbaru' (indeks K-gram)")
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    tr = raw[:a.ntrain]
    held = raw[a.ntrain:a.ntrain + a.nheld]
    V = 256
    print(f"data={a.data} ntrain={len(tr)} nheld={len(held)} d={a.d} L={a.L} "
          f"lr={a.lr} mixer={a.mixer} zero_out={a.zero_out} init={a.mixer_init} "
          f"mult={a.mixer_mult} seed={a.seed}")

    cfg = Config.best(d=a.d, n_layers=a.L, vocab=V, d_mem=max(8, a.d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0,
                      mixer_k=(1 if a.mixer > 0 else 0),
                      mixer_match_orders=",".join(str(i) for i in range(1, a.mixer + 1)),
                      mixer_init=a.mixer_init, mixer_lr_mult=a.mixer_mult,
                      mixer_match=bool(a.match),
                      mixer_match_recent=bool(a.match_recent),
                      out_zero_init=bool(a.zero_out),
                      think_max=0, use_innovation=False, seg_len=10 ** 9, lr=a.lr, seed=a.seed)
    m = TMT2(cfg)
    fused = (a.mixer == 0)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=a.seed, use_stop=False)
    L = FastLearner(m, o)
    s = Stream(m, online=True)

    hdr = f"{'epoch':>5}{'CE latih':>12}{'CE tahan':>12}{'detik':>9}"
    print(hdr)
    print("-" * len(hdr))
    for ep in range(a.epochs):
        t0 = time.time()
        ce_tr = []
        for i in range(len(tr)):
            c = L.step_fused(s, int(tr[i]), train=True) if fused else L.step(s, int(tr[i]), train=True)
            if c is not None:
                ce_tr.append(c)
        ce_h = float(L.eval_stream([int(b) for b in held]).mean() / LN2)
        print(f"{ep+1:>5}{float(np.mean(ce_tr)/LN2):>12.4f}{ce_h:>12.4f}{time.time()-t0:>9.1f}",
              flush=True)


if __name__ == "__main__":
    main()
