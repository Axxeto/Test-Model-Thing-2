"""Ablasi multi-seed: bandingkan komponen TMT2 pada aliran byte yang sama.

  python scripts/ablate.py --data periodic --n 4000 --seeds 4
  python scripts/ablate.py --data license  --n 6000 --seeds 3 --variants default,no_think,no_replay

Laporan: bits/byte prekuensial pada seperempat TERAKHIR aliran (rata-rata ± simpangan baku antar seed).
Satu seed tidak cukup untuk mengklaim komponen berguna: variasi antar seed pada aliran pendek besar.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from multiprocessing import Pool
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Learner, Stream, TrainOpts
from tmt2.corpus import load

VARIANTS = {
    "default": ({}, {}),
    "no_think": ({}, dict(think_prob=0.0)),
    "no_replay": (dict(replay=False), {}),
    "no_memory": (dict(use_memory=False), {}),
    "no_innovation": (dict(use_innovation=False), {}),
    "no_pcgrad": (dict(pcgrad=False), {}),
    "no_uncertainty": (dict(uncertainty=False), {}),
}


def get_data(kind, n):
    return load(kind)[:n]                # lintas-platform (data/licenses/ atau /usr/share/...)


def run_one(job):
    name, seed, kind, n, d, lr, slow_every = job
    ckw, tkw = VARIANTS[name]
    cfg = Config(d=d, n_layers=3, d_mem=max(8, d // 4), lr=lr, slow_every=slow_every, seed=seed, **ckw)
    m = TMT2(cfg)
    m.logit_scale.data = np.array(np.log(3.0))
    ln = Learner(m, TrainOpts(seed=seed, **tkw))
    s, bits = Stream(m), []
    for b in get_data(kind, n):
        c = ln.step(s, b)
        if c is not None:
            bits.append(c / np.log(2))
    bits = np.array(bits)
    return name, seed, bits[-len(bits) // 4:].mean(), bits.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="periodic", choices=["periodic", "license"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--slow_every", type=int, default=8)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    names = a.variants.split(",")
    jobs = [(nm, sd, a.data, a.n, a.d, a.lr, a.slow_every) for nm in names for sd in range(a.seeds)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    print(f"data={a.data} n={a.n} d={a.d} seeds={a.seeds}  ({time.time()-t0:.0f} detik)")
    print(f"{'varian':<16}{'seperempat akhir (bpb)':>26}{'semua (bpb)':>16}")
    for nm in names:
        last = np.array([r[2] for r in res if r[0] == nm]); allb = np.array([r[3] for r in res if r[0] == nm])
        print(f"{nm:<16}{last.mean():>17.3f} ± {last.std():.3f}{allb.mean():>16.3f}")


if __name__ == "__main__":
    main()
