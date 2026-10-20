"""Ukur KUALITAS GENERASI secara TIDAK SIRKULAR (menjawab telaah eksternal, §17).

Masalah pada `gen_quality.py` (versi lama):
  * Penilai n-gram-3 pada output model BERBASIS n-gram = sirkular (menilai diri sendiri).
  * Skor rendah otomatis didapat oleh teks repetitif bersuhu rendah.
  * Tidak ada ukuran REPETISI ANTAR-SAMPEL (mode kolaps / penarik hafalan).
  * Pembanding "mixer vs SSM-saja" tidak membuktikan nilai tambah TMT2 (mixer memang n-gram);
    pembanding yang SAH = PPM orde-8 murni pada split yang sama.

Metrik baru (semuanya objektif, tanpa penilai n-gram sebagai hakim):
  1. **overlap antar-sampel**: panjang substring bersama TERPANJANG antara pasangan sampel
     dari prompt BERBEDA. Tinggi = model jatuh ke penarik hafalan (bukan generalisasi).
  2. **distinct-n** (n=1,2,3,4): fraksi n-gram unik; rendah = repetitif.
  3. **rasio kata-valid**: fraksi token (dipisah spasi) yang muncul di kosakata data UJI TAHAN.
     Random -> ~0; bahasa asli -> tinggi.
  4. **bpb uji-tahan jujur** + pembanding **PPM orde-8 murni** (OnlineNgramN) pada split sama.
  5. (opsional) keragaman lintas seed: Jaccard kata antar generasi seed berbeda.
"""
import argparse, json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner, OnlineNgramN

LN2 = float(np.log(2))


def lcs_len(a, b):
    """Panjang substring bersama terpanjang (dua baris, O(len a * len b))."""
    a = list(a); b = list(b)
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best


def distinct_n(text, n):
    toks = text  # byte-level: pakai byte sebagai token
    if len(toks) < n:
        return 0.0
    grams = Counter(bytes(toks[i:i + n]) for i in range(len(toks) - n + 1))
    total = len(toks) - n + 1
    return len(grams) / max(total, 1)


def valid_word_ratio(text, vocab):
    """Fraksi token (potong spasi) yang ada di kosakata data uji tahan."""
    parts = bytes(text).split(b" ")
    if not parts:
        return 0.0
    ok = sum(1 for p in parts if p and p in vocab)
    return ok / len(parts)


def build(seed, d, layers, use_mixer, orders):
    if use_mixer:
        cfg = Config.best(vocab=256, d=d, n_layers=layers, d_mem=max(8, d // 4), slow_every=8,
                          seg_len=256, replay=False, sleep_every=10 ** 9, seed=seed)
        cfg.mixer_match_orders = orders
    else:
        cfg = Config(vocab=256, d=d, n_layers=layers, d_mem=max(8, d // 4), head="linear",
                     deep_grad=True, lr=1e-2, warmup=200, slow_every=8, seg_len=256,
                     replay=False, sleep_every=10 ** 9, mixer_k=0, seed=seed)
    m = TMT2(cfg)
    fl = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, only_byte_loss=True, use_stop=False))
    return m, fl


def ppm_order8(test_probe_prefix, test_bytes, order=8):
    """Baseline adil: PPM/ngram orde-8 MURNI dilatih pada train, dieval prekuensial di test."""
    return None  # dihitung di main dengan OnlineNgramN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--ntrain", type=int, default=300000)
    ap.add_argument("--ntest", type=int, default=30000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n_gen", type=int, default=1200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--orders", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--out", default="results/gen_quality_v2.jsonl")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    raw = open(a.data, "rb").read()
    train_raw = raw[:a.ntrain]
    test_raw = raw[a.ntrain:a.ntrain + a.ntest]
    vocab = set(test_raw.split())                 # kosakata data UJI TAHAN (bukan data latih)
    print(f"data {a.data}: latih {a.ntrain}B, uji-tahan {a.ntest}B, vocab-uji={len(vocab)} kata")

    # ---------- baseline PPM orde-8 murni pada split yang sama ----------
    ng = OnlineNgramN(256, 8)
    hist = []
    for b in train_raw:
        ng.update(tuple(hist)[-8:], int(b)); hist.append(int(b))
    ht = list(hist[-8:])
    ce, n = 0.0, 0
    for b in test_raw:
        lp = ng.logprior_order(tuple(ht), 8)
        ce += -lp[int(b)] / LN2
        n += 1
        ht.append(int(b)); ht = ht[-8:]
    ppm8 = ce / max(n, 1)
    print(f"baseline PPM orde-8 murni (latih=train, eval beku test) = {ppm8:.4f} bit")

    # byte acak sebagai jangkar bawah untuk metrik kata
    rng0 = np.random.default_rng(0)
    rand_bytes = bytes(rng0.integers(0, 256, a.n_gen, dtype=np.uint8))
    print(f"jangkar: kata-valid byte-acak={valid_word_ratio(rand_bytes, vocab):.3f}")

    rows = []
    for use_mixer in (True, False):
        tag = "mixer" if use_mixer else "SSM-saja"
        for seed in seeds:
            m, fl = build(seed, a.d, a.layers, use_mixer, a.orders)
            t0 = time.time()
            s = Stream(m, online=True)
            for b in np.frombuffer(train_raw, dtype=np.uint8).astype(np.int64):
                fl.step(s, int(b), train=True, think=False)
            ce, n = 0.0, 0
            for b in test_raw:
                c = fl.step(s, int(b), train=False, think=False)
                if c is not None:
                    ce += float(c); n += 1
            test_bpb = ce / max(n, 1) / LN2
            # generasi dari TIGA prompt BERBEDA -> ukur overlap antar-sampel
            probes = [bytes(test_raw[:60]), bytes(test_raw[500:560]), bytes(test_raw[1500:1560])]
            gens = [bytes(fl.generate(p, a.n_gen, temp=0.7, think=False, seed=seed + i))
                    for i, p in enumerate(probes)]
            # 1. overlap antar-sampel: LCS pasangan maksimum
            lcs = max(lcs_len(gens[0], gens[1]), lcs_len(gens[0], gens[2]), lcs_len(gens[1], gens[2]))
            # 2. distinct-n rata-rata lintas sampel
            dn = {f"d{n_}": round(np.mean([distinct_n(g, n_) for g in gens]), 4) for n_ in (1, 2, 3, 4)}
            # 3. rasio kata-valid rata-rata
            vwr = float(np.mean([valid_word_ratio(g, vocab) for g in gens]))
            # 4. Jaccard kata antar-seed (dikumpulkan setelah semua seed; placeholder)
            row = dict(variant=tag, seed=seed, test_bpb=round(test_bpb, 4),
                       cross_lcs=lcs, **dn, valid_word_ratio=round(vwr, 4),
                       train_s=round(time.time() - t0, 1), ppm_order8=round(ppm8, 4))
            rows.append(row)
            print(f"  {tag:9s} seed={seed} bpb_uji={test_bpb:.4f}  LCS-antar-sampel={lcs:4d}  "
                  f"D1={dn['d1']:.3f} D4={dn['d4']:.3f}  kata-valid={vwr:.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\n->", a.out)
    print("Bacaan: LCS-antar-sampel RENDAH + D4 TINGGI + kata-valid TINGGI = generalisasi sehat.")
    print("        LCS TINGGI menandakan penarik hafalan lintas prompt (mode kolaps).")


if __name__ == "__main__":
    main()
