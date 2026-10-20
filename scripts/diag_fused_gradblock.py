"""Lokalisasi blok gradien yang berbeda (fused vs lama) pada byte yang divergen."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner


def make(use_memory, fused):
    cfg = Config.best(d=16, n_layers=2, vocab=64, d_mem=8, use_memory=use_memory,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0, use_stop=False)
    return cfg, m, FastLearner(m, o)


def main():
    use_memory = len(sys.argv) < 2 or sys.argv[1] != "nomem"
    data = [3, 17, 42, 5]
    cfgR, mR, LR = make(use_memory, False)
    cfgF, mF, LF = make(use_memory, True)
    sR = Stream(mR, online=True); sF = Stream(mF, online=True)

    # peta nama -> (offset, panjang)
    blocks = []
    for name, (off, shp) in LF.offs.items():
        n = int(np.prod(shp)) if len(shp) else 1
        blocks.append((name, off, n))

    for i, b in enumerate(data):
        LR.step(sR, b, train=True)
        LF.step_fused(sF, b, train=True)
        gA = LR.gacc.get("byte"); gB = LF.gacc.get("byte")
        if gA is None or gB is None:
            print(f"byte {i}: no grad"); continue
        rows = []
        for name, off, n in blocks:
            dv = float(np.abs(gA[off:off + n] - gB[off:off + n]).max())
            if dv > 1e-12:
                rows.append((dv, name))
        rows.sort(reverse=True)
        print(f"byte {i}: max={np.abs(gA-gB).max():.3e}  blocks={[(n, f'{v:.2e}') for v,n in rows[:8]]}")


if __name__ == "__main__":
    main()
