"""Bandingkan STATE setelah SATU byte maju (dari state nol), fused vs lama."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2, l2n_np
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner


def make(seed=0):
    cfg = Config.best(d=16, n_layers=2, vocab=64, d_mem=8, use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    return cfg, m


def main():
    cfg, m = make()
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=0, use_stop=False)
    LB = FastLearner(m, o)
    o2 = TrainOpts(only_byte_loss=True, use_fused_kernel=False, seed=0, use_stop=False)
    LA = FastLearner(m, o2)

    b = 17
    x = m.E.data[b].copy()
    # ---- lama: panggil _stack_step langsung dari state nol
    st = LA.m.layers[0] and None
    from tmt2.core import StackState
    stA = StackState(m)
    recs, newA = LA._stack_step(x, stA, None)
    # ---- fused
    fp_out = LB.f_out
    from tmt2 import fastpath as fp
    fp.fused_fwd(LB.f_bundle, LB.f_st, LB.f_loff, x, fp_out, LB.f_scr)
    print("state fused selesai")
    # bandingkan
    for l in range(2):
        base = l * LB.f_ssz
        h_f = LB.f_st[base:base + 16]
        h_o = newA.layers[l].h
        y_f = LB.f_st[base + 16 + 3 * 256 + 16:base + 16 + 3 * 256 + 32]
        y_o = recs[l].y
        SWa_f = LB.f_st[base + 16:base + 16 + 256].reshape(16, 16)
        SWa_o = newA.layers[l].S["Wa"]
        print(f"L{l} h   {np.abs(h_f-h_o).max():.3e}")
        print(f"L{l} y   {np.abs(y_f-y_o).max():.3e}")
        print(f"L{l} SWa {np.abs(SWa_f-SWa_o).max():.3e}")
        Mf = LB.f_st[base+16+768+32:base+16+768+32+64].reshape(8,8)
        print(f"L{l} M   {np.abs(Mf-newA.layers[l].M).max():.3e}")
        kp_f = LB.f_st[base+16+768+32+64:base+16+768+32+64+16]
        print(f"L{l} kp  {np.abs(kp_f-newA.layers[l].kp).max():.3e}")


if __name__ == "__main__":
    main()
