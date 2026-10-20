"""Bandingkan gradien SETELAH TIAP byte (fused vs lama) untuk menemukan langkah yang beda."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner


def make(fused):
    cfg = Config.best(d=16, n_layers=2, vocab=64, d_mem=8, use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0, use_stop=False)
    return m, FastLearner(m, o)


def gvec(L):
    g = L.gacc.get("byte")
    return None if g is None else g.copy()


def main():
    data = [3, 17, 42, 5]
    mA, LA = make(False); sA = Stream(mA, online=True)
    mB, LB = make(True); sB = Stream(mB, online=True)
    for i, b in enumerate(data):
        LA.step(sA, b, train=True)
        LB.step_fused(sB, b, train=True)
        gA, gB = gvec(LA), gvec(LB)
        if gA is None or gB is None:
            print(f"byte {i}: gA={'None' if gA is None else 'ok'} gB={'None' if gB is None else 'ok'}")
            continue
        diff = np.abs(gA - gB)
        print(f"byte {i}: maxdiff {diff.max():.3e}  sumA {gA.sum():.6f} sumB {gB.sum():.6f}  #nz {np.count_nonzero(diff>1e-9)}")


if __name__ == "__main__":
    main()
