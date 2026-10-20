"""UKUR NORMA & KEPENUHAN GERBANG pada TEKS NYATA (bukan data acak yang dulu rusak).

Pertanyaan: apakah ``y`` meledak dan apakah gerbang ``a = sigmoid(...)`` jenuh (a -> 1)?
Meledaknya norma adalah dugaan B4 dan alasan diusulkannya ``pre_norm``. Sebelum menulis
``pre_norm``, ukur dulu di WikiText agar tidak memperbaiki masalah yang tidak ada.

Melaporkan, per lapisan, selama lintasan latih:
  * log10 |y| (min/median/maks)
  * fraksi gerbang a > 0.99 (jenuh) dan < 0.01
  * log10 |h|
Gerbang ``a`` dihitung ulang dari parameter yang sama seperti kernel: a = sigmoid(x @ Wa + ba).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load, sanity_check


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="wikitext")
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--d_mem", type=int, default=8)
    ap.add_argument("--se", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--deep", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jsonl", default="results/norm_real.jsonl")
    a = ap.parse_args()

    raw = load(a.kind, n=a.n)
    sanity_check(raw, name=f"norm:{a.kind}", expect_len=a.n)
    cfg = Config(d=a.d, n_layers=a.L, d_mem=a.d_mem, head="linear", deep_grad=bool(a.deep),
                 lr=a.lr, warmup=200, slow_every=a.se, clip=1.0, wd=0.0, replay=False,
                 adapter_rank=0, use_innovation=False, mem_query_x=True, mem_wq_match_k=True,
                 mem_hetero=True, mem_surprise_fixed=True, beta_max=1.0, seed=a.seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=a.seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)

    # akumulator per lapisan
    ys = {l: [] for l in range(a.L)}
    hs = {l: [] for l in range(a.L)}
    sat = {l: 0 for l in range(a.L)}
    tot = 0
    step_every = max(1, a.n // 40)
    print(f"korpus={a.kind} n={a.n} d={a.d} L={a.L} epochs={a.epochs}")
    for ep in range(a.epochs):
        for i, b in enumerate(raw):
            ln.step(s, int(b))
            if i % step_every:
                continue
            # inspeksi state tiap lapisan
            for l in range(a.L):
                st = s.st.layers[l]
                y = st.rec.y if st.rec is not None else None
                if y is not None:
                    ny = float(np.linalg.norm(np.asarray(y)))
                    nh = float(np.linalg.norm(np.asarray(st.h)))
                    if ny > 0:
                        ys[l].append(np.log10(ny + 1e-30))
                    if nh > 0:
                        hs[l].append(np.log10(nh + 1e-30))
                Wa = np.asarray(m.params[f"L{l}.Wa"].data)
                ba = np.asarray(m.params[f"L{l}.ba"].data)
                xvec = np.asarray(m.encode_byte(int(b)).data)
                if Wa.shape[0] == len(xvec):
                    av = sigmoid(xvec @ Wa + ba)
                    sat[l] += int(np.sum(av > 0.99))
                    tot += 1

    rows = []
    print(f"\n{'lapisan':>8}{'log10|y| med':>14}{'maks':>8}{'log10|h| med':>14}{'frac a>0.99':>13}")
    print("-" * 60)
    for l in range(a.L):
        ym = np.median(ys[l]) if ys[l] else float("nan")
        yx = np.max(ys[l]) if ys[l] else float("nan")
        hm = np.median(hs[l]) if hs[l] else float("nan")
        fr = sat[l] / max(tot, 1)
        print(f"{l:>8}{ym:>14.2f}{yx:>8.2f}{hm:>14.2f}{fr:>13.3f}")
        rows.append(dict(kind=a.kind, n=a.n, layer=l, log10_y_med=float(ym), log10_y_max=float(yx),
                         log10_h_med=float(hm), frac_a_sat=fr))
    alls = [r["log10_y_max"] for r in rows]
    print(f"\nlog10|y| maks tertinggi = {max(alls):.2f}  (ledakan bila > 5-6)")
    print("frac a>0.99 tinggi = gerbang jenuh (integrasi tanpa peluruhan).")
    os.makedirs(os.path.dirname(a.jsonl) or ".", exist_ok=True)
    with open(a.jsonl, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
