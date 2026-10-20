"""Diagnosa per-byte: bandingkan arena fused vs StackState referensi per sub-array."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2 import fastpath as fp


def make(use_memory, fused):
    cfg = Config.best(d=16, n_layers=2, vocab=64, d_mem=8, use_memory=use_memory,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0, use_stop=False)
    return cfg, m, FastLearner(m, o)


def dump_ref(st):
    out = {}
    for l, ls in enumerate(st.layers):
        out[(l, "h")] = ls.h.copy()
        out[(l, "M")] = ls.M.copy()
        out[(l, "kp")] = ls.kp.copy()
        for k in ("Wa", "B", "K", "ba"):
            out[(l, k)] = ls.S[k].copy()
    return out


def dump_arena(L):
    d, dm, nl = L.f_d, L.f_dm, L.f_nl
    _, so = fp.state_offsets(nl, d, dm)
    out = {}
    for l in range(nl):
        base = l * L.f_ssz
        for name, key in (("h", "h"), ("Wa", "SWa"), ("B", "SB"), ("K", "SK"),
                          ("ba", "Sba"), ("M", "M"), ("Mpost", "Mpost"), ("kp", "kp")):
            o = base + so[key]
            if name in ("M", "Mpost"):
                arr = L.f_st[o:o + dm * dm].reshape(dm, dm)
            elif name in ("Wa", "B", "K"):
                arr = L.f_st[o:o + d * d].reshape(d, d)
            else:
                arr = L.f_st[o:o + d]
            out[(l, name)] = arr.copy()
    return out


def main():
    use_memory = len(sys.argv) < 2 or sys.argv[1] != "nomem"
    data = [3, 17, 42, 5]
    cfgR, mR, LR = make(use_memory, False)
    cfgF, mF, LF = make(use_memory, True)
    sR = Stream(mR, online=True); sF = Stream(mF, online=True)

    for i, b in enumerate(data):
        LR.step(sR, b, train=True)
        LF.step_fused(sF, b, train=True)
        dA = dump_ref(sR.st)
        dB = dump_arena(LF)
        worst = []
        for l in range(LF.f_nl):
            for name in ("h", "Wa", "B", "K", "ba", "M", "kp"):
                A = dA.get((l, name)); B = dB.get((l, name))
                if A is None or B is None:
                    continue
                dv = float(np.abs(A - B).max())
                if dv > 1e-12:
                    worst.append(f"L{l}.{name}={dv:.2e}")
        print(f"byte {i}: " + ("OK" if not worst else "DIFF " + " ".join(worst)))


if __name__ == "__main__":
    main()
