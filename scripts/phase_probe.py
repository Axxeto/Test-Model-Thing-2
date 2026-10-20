"""CE per fase: langkah-global mod P vs posisi-dalam-epoch mod P, plus hitungan
eksekusi cabang byte dan norma |glo| per fase.

Tujuan: membedakan "jam di kode pelatihan" dari "sifat data/optimasi".
Petunjuk utama = apakah pola mengikuti langkah GLOBAL atau POSISI-dalam-epoch.
Kunci: panjang data yang BUKAN kelipatan P (mis. 127/131) memisahkan keduanya.

  python scripts/phase_probe.py --impl fast --n 128 --epochs 80
  python scripts/phase_probe.py --impl fast --n 131                 # buka kunci data
  python scripts/phase_probe.py --impl ref  --n 128 --layers 1      # autograd
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Learner, Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2 import fastpath as fp

LN2 = float(np.log(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="fast", choices=["fast", "ref"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--V", type=int, default=64)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--slow_every", type=int, default=16)
    ap.add_argument("--s_decay", type=float, default=0.5)
    ap.add_argument("--vic_every", type=int, default=8)
    ap.add_argument("--seg_len", type=int, default=32)
    ap.add_argument("--sleep_every", type=int, default=512)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--only_byte", action="store_true")
    ap.add_argument("--d_mem", type=int, default=0)
    ap.add_argument("--no_mem", action="store_true")
    ap.add_argument("--period", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    P = a.period
    data = np.random.default_rng(a.seed).integers(0, a.V, a.n)
    cfg = Config(d=a.d, n_layers=a.layers, vocab=a.V, d_mem=a.d_mem or max(8, a.d // 4),
                 head="linear", deep_grad=a.layers > 1, lr=a.lr, warmup=a.warmup,
                 slow_every=a.slow_every, s_decay=a.s_decay, replay=a.replay, seg_len=a.seg_len,
                 sleep_every=a.sleep_every, adapter_rank=0, use_innovation=False,
                 use_memory=not a.no_mem, seed=a.seed)
    m = TMT2(cfg)
    K = FastLearner if a.impl == "fast" else Learner
    ln = K(m, TrainOpts(seed=a.seed, think_prob=0.0, use_stop=False,
                        only_byte_loss=a.only_byte, vic_every=a.vic_every))
    s = Stream(m)

    # jejak |glo| dan JUMLAH eksekusi cabang byte per fase langkah-global
    cur = {"t": -1}
    glo_rec = [[] for _ in range(P)]
    if a.impl == "fast":
        orig_step, orig_hb = ln.step, fp.head_bwd_lin_acc

        def step(st, b, *x, **k):
            cur["t"] = st.t + 1
            return orig_step(st, b, *x, **k)

        def hb(p, W, glo, w, Pm, y, g, oW):
            glo_rec[cur["t"] % P].append(float(np.linalg.norm(glo)))
            return orig_hb(p, W, glo, w, Pm, y, g, oW)
        ln.step, fp.head_bwd_lin_acc = step, hb

    gs, gc = np.zeros(P), np.zeros(P)   # fase = langkah GLOBAL (indeks byte target)
    ps, pc = np.zeros(P), np.zeros(P)   # fase = POSISI dalam epoch
    t = 0
    for ep in range(a.epochs):
        for i, b in enumerate(data):
            c = ln.step(s, int(b))
            if c is not None and ep >= a.epochs - a.last:
                v = c / LN2
                gs[t % P] += v; gc[t % P] += 1
                ps[i % P] += v; pc[i % P] += 1
            t += 1
    if a.impl == "fast":
        ln.step, fp.head_bwd_lin_acc = orig_step, orig_hb

    ev = ln.eval_stream([int(b) for b in data]) / LN2   # indeks j -> byte posisi j+1
    es, ec = np.zeros(P), np.zeros(P)
    for j, v in enumerate(ev):
        es[(j + 1) % P] += v; ec[(j + 1) % P] += 1

    print(f"impl={a.impl} n={a.n} V={a.V} L={a.layers} slow_every={a.slow_every} "
          f"vic_every={a.vic_every} replay={a.replay} only_byte={a.only_byte} "
          f"mem={not a.no_mem}  seragam={np.log2(a.V):.2f} bit")
    print(f"{'fase':>5}{'latih/global':>14}{'latih/posisi':>14}{'beku/posisi':>13}"
          f"{'|glo| rata':>12}{'#glo':>7}")
    for k in range(P):
        gl = glo_rec[k]
        print(f"{k:>5}{gs[k]/max(gc[k],1):>14.3f}{ps[k]/max(pc[k],1):>14.3f}"
              f"{es[k]/max(ec[k],1):>13.3f}"
              f"{(np.mean(gl) if gl else float('nan')):>12.3f}{len(gl):>7}")
    print(f"\nRINGKAS: beku/posisi  min={es.min():.3f} maks={es.max():.3f} "
          f"selisih={es.max()-es.min():.3f}   #glo min={min(len(x) for x in glo_rec)} "
          f"maks={max(len(x) for x in glo_rec)}")


if __name__ == "__main__":
    main()
