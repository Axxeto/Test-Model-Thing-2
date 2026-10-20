"""Benchmark end-to-end: latih N byte lewat jalur lama vs jalur fused, ukur B/s.

Pakai korpus nyata (license) dan konfigurasi kompatibel fused. Protokol adil: model BARU
per varian (seed sama), pemanasan sebelum ukur. Bandingkan juga BPB akhir (harus sama).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts, ce_np
from tmt2.fast_learner import FastLearner


def run(fused, data, d=32, nl=2, V=256, nbytes=8000, slow_every=256):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=max(8, d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False, seg_len=10 ** 9)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0, use_stop=False)
    L = FastLearner(m, o)
    s = Stream(m, online=True)
    # warmup 300 byte (compile + state stabil)
    for i in range(300):
        b = int(data[i]) % V
        if fused:
            L.step_fused(s, b, train=True)
        else:
            L.step(s, b, train=True)
    # reset statistik
    L.n_bytes = 0
    L.t_adam = 0
    seg_ce, seg_n = 0.0, 0
    t0 = time.perf_counter()
    for i in range(nbytes):
        b = int(data[300 + i]) % V
        if fused:
            c = L.step_fused(s, b, train=True)
        else:
            c = L.step(s, b, train=True)
        if c is not None:
            seg_ce += c
            seg_n += 1
    dt = time.perf_counter() - t0
    bpb = (seg_ce / seg_n) / np.log(2) if seg_n else float("nan")
    return nbytes / dt, bpb


def main():
    data = np.frombuffer(load("license"), dtype=np.uint8)
    for d in (32, 64):
        r_old = run(False, data, d=d)
        r_new = run(True, data, d=d)
        print(f"d={d}:  lama {r_old[0]:7.0f} B/s (BPB {r_old[1]:.4f})  |  "
              f"fused {r_new[0]:7.0f} B/s (BPB {r_new[1]:.4f})  |  speedup {r_new[0]/r_old[0]:.2f}x")


if __name__ == "__main__":
    main()
