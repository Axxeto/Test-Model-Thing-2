"""Ukur laju byte murni jalur fused: min-of-B/byte (tahan derau beban mesin).

Tidak ada slow_update, tidak ada debug. Ambil median dari beberapa ronde; juga laporkan min
per-byte utk melihat lantai sebenarnya tanpa gangguan penjadwal.
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

P = time.perf_counter


def main():
    data = np.frombuffer(load("license"), dtype=np.uint8)
    for d in (32, 64):
        V, nl = 256, 2
        cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=max(8, d // 4), use_memory=True,
                          head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                          think_max=0, use_innovation=False, seg_len=10 ** 9,
                          slow_every=10 ** 9)         # MATI: ukur murni jalur byte
        m = TMT2(cfg)
        o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=0, use_stop=False)
        L = FastLearner(m, o)
        s = Stream(m, online=True)
        for i in range(500):
            L.step_fused(s, int(data[i]) % V, train=True)
        ro = []
        for rnd in range(5):
            n = 2000
            t0 = P()
            for i in range(n):
                L.step_fused(s, int(data[500 + i]) % V, train=True)
            ro.append(n / (P() - t0))
        ro.sort()
        print(f"d={d}: median {ro[len(ro)//2]:7.0f} B/s   min {ro[0]:7.0f}   max {ro[-1]:7.0f}  "
              f"(5x2000 byte)")


if __name__ == "__main__":
    main()
