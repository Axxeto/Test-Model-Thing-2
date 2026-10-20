"""Ukur KUALITAS GENERASI secara objektif (bukan mata manusia).

Masalah: "generasi terlihat bagus" itu subjektif. Kita butuh angka.

Metrik yang dipakai:
  1. **BPB uji-tahan (held-out)**: latih pada prefix, ukur CE pada SISA data (tidak pernah
     dilihat). Ini ukuran model bahasa yang jujur.
  2. **Skor generasi oleh penilai luar (held-out n-gram)**: latih model n-gram orde-3 pada
     data UJI TAHAN (bukan data latih TMT2), lalu ukur berapa buruk/bagus teks GENERASI
     menurut penilai itu. Teks acak -> skor tinggi (buruk). Teks mirip bahasa Indonesia asli
     -> skor rendah (bagus).
     Baseline pembanding: teks ASLI dari data uji (skor "lantai" yang bisa dicapai), dan
     byte acak seragam (skor "langit-langit" buruk).

Ini memisahkan "model menghasilkan teks yang mirip bahasa" dari "model menghasilkan sampah".

  python scripts/gen_quality.py --data data/indonesian_1m_pure.txt --ntrain 300000 --ntest 30000
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


class NgramScorer:
    """N-gram orde-N (Witten-Bell, back-off ke unigram) sebagai PENILAI LUAR."""

    def __init__(self, order=3):
        self.order = order
        self.uni = np.ones(256)
        self.tabs = [dict() for _ in range(order)]

    def fit(self, data):
        for i, b in enumerate(data):
            b = int(b)
            self.uni[b] += 1.0
            for k in range(1, self.order + 1):
                if i < k:
                    break
                ctx = bytes(data[i - k:i])
                c = self.tabs[k - 1].setdefault(ctx, [{}, 0.0])
                c[0][b] = c[0].get(b, 0.0) + 1.0
                c[1] += 1.0
        return self

    def logp(self, ctx, b):
        p = self.uni / self.uni.sum()
        for k in range(1, self.order + 1):
            if len(ctx) < k:
                break
            c = self.tabs[k - 1].get(ctx[-k:])
            if c is None:
                break
            d, n = c
            lam = n / (n + len(d))
            p = (1.0 - lam) * p
            for tok, cnt in d.items():
                p[tok] += lam * cnt / n
        return float(np.log(p[int(b)] + 1e-12))

    def score(self, text):
        """bit/byte rata-rata teks menurut penilai (rendah = bagus)."""
        s, n = 0.0, 0
        for i, b in enumerate(text):
            if i == 0:
                continue
            s += -self.logp(bytes(text[max(0, i - self.order):i]), b)
            n += 1
        return s / max(n, 1) / LN2


def build(seed, d, layers, use_mixer):
    if use_mixer:
        cfg = Config.best(vocab=256, d=d, n_layers=layers, d_mem=max(8, d // 4), slow_every=8,
                          seg_len=256, replay=False, sleep_every=10 ** 9, seed=seed)
    else:
        cfg = Config(vocab=256, d=d, n_layers=layers, d_mem=max(8, d // 4), head="linear",
                     deep_grad=True, lr=1e-2, warmup=200, slow_every=8, seg_len=256,
                     replay=False, sleep_every=10 ** 9, mixer_k=0, seed=seed)
    m = TMT2(cfg)
    fl = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, only_byte_loss=True, use_stop=False))
    return m, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--ntrain", type=int, default=300000)
    ap.add_argument("--ntest", type=int, default=30000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n_gen", type=int, default=2000, help="byte yang digenerasi per varian")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gen_quality.jsonl")
    a = ap.parse_args()

    raw = open(a.data, "rb").read()
    train_ids = np.frombuffer(raw[:a.ntrain], dtype=np.uint8).astype(np.int64)
    test_raw = raw[a.ntrain:a.ntrain + a.ntest]
    print(f"data {a.data}: latih {a.ntrain}B, uji-tahan {a.ntest}B")

    # penilai luar: n-gram orde-3 HANYA dari data uji tahan (bukan data latih TMT2)
    scorer = NgramScorer(3).fit(test_raw)
    real_score = scorer.score(test_raw)
    rng = np.random.default_rng(0)
    rand_score = scorer.score(bytes(rng.integers(0, 256, 2000, dtype=np.uint8)))
    print(f"penilai n-gram-3: teks ASLI uji = {real_score:.3f} bit  |  byte ACAK = {rand_score:.3f} bit")

    rows = []
    for use_mixer in (True, False):
        tag = "mixer(Config.best)" if use_mixer else "SSM-saja"
        m, fl = build(a.seed, a.d, a.layers, use_mixer)
        t0 = time.time()
        s = Stream(m, online=True)
        for b in train_ids:
            fl.step(s, int(b), train=True, think=False)
        # BPB UJI-TAHAN (prekuensial, bobot beku, state dibawa dari latih)
        ce, n = 0.0, 0
        for b in test_raw:
            c = fl.step(s, int(b), train=False, think=False)
            if c is not None:
                ce += float(c); n += 1
        test_bpb = ce / max(n, 1) / LN2
        # GENERASI lalu nilai oleh penilai luar
        prompt = bytes(raw[:60])
        gen = fl.generate(prompt, a.n_gen, temp=0.7, think=False, seed=a.seed)
        gen_b = bytes(gen)
        g_score = scorer.score(gen_b)
        uniq = len(set(gen_b)) / max(len(gen_b), 1)
        row = dict(variant=tag, test_bpb=round(test_bpb, 4), gen_judge_bit=round(g_score, 4),
                   gen_uniq_frac=round(uniq, 4), train_s=round(time.time() - t0, 1),
                   real_judge_bit=round(real_score, 4), rand_judge_bit=round(rand_score, 4))
        rows.append(row)
        print(f"  {tag:22s} bpb_uji={test_bpb:.4f}  skor-generasi={g_score:.3f} bit  "
              f"(asli {real_score:.3f} / acak {rand_score:.3f})  uniq={uniq:.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import json
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("->", a.out)
    print("\nBacaan: skor-generasi mendekati 'asli' = teks mirip Indonesia; mendekati 'acak' = sampah.")


if __name__ == "__main__":
    main()
