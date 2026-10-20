"""AUDIT: berapa byte yang SEBENARNYA dievaluasi, dan apakah CE per fase masih
muncul ketika n BUKAN kelipatan 8?

Skrip ini sengaja bodoh dan eksplisit: mencetak len(data), jumlah langkah, dan
CE per fase untuk beberapa n. Tujuannya menutup keraguan metodologis dari
telaah eksternal (butir 2b/2c) dan kesalahan panjang di probe sebelumnya.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


def run(n, V=64, d=32, L=3, lr=3e-2, epochs=80, se=16, P=8, seed=7, tag=""):
    data = np.random.default_rng(seed).integers(0, V, n)
    cfg = Config(d=d, n_layers=L, d_mem=8, head="linear", deep_grad=False, lr=lr,
                 warmup=200, slow_every=se, clip=1.0, wd=0.0, replay=False, adapter_rank=0,
                 use_innovation=False, mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)
    for _ in range(epochs):
        for b in data:
            ln.step(s, int(b))
    # evaluasi eksplisit: SATU lintasan, hitung CE per fase langkah-global
    st = Stream(m, online=False)
    ce = []
    for i, b in enumerate(data):
        c = ln.step(st, int(b), train=False, think=False)
        ce.append(np.nan if c is None else c / LN2)
    ce = np.array(ce)
    assert len(ce) == n, f"len(ce)={len(ce)} != n={n}"
    ph = np.array([np.nanmean(ce[k::P]) for k in range(P)])
    print(f"{tag:<22} n={n:>4} len(data)={len(data)} langkah={len(ce)}  "
          f"selisih-fase={ph.max()-ph.min():>7.3f}  rata2={np.nanmean(ce):.3f}")
    print("   " + " ".join(f"f{k}={v:.3f}" for k, v in enumerate(ph)))
    return ph


def main():
    print("AUDIT panjang & fase — V=64, d=32, L=3, lr=3e-2, 80 epoch, only_byte")
    print("(n kelipatan 8 -> 'langkah global mod 8' == 'posisi dalam epoch mod 8')")
    print()
    for n in (120, 128, 131, 135, 200, 257):
        run(n, tag="n=%d" % n)
    print()
    print("INTERPRETASI:")
    print("  bila selisih-fase besar HANYA pada n kelipatan 8 -> aliasing, bukan jam kode.")
    print("  bila besar di semua n -> sifat optimasi/data yang nyata.")


if __name__ == "__main__":
    main()
