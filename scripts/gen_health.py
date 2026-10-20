"""Kesehatan generasi Tier A: metrik non-sirkular M1-M7 (#2, T0).

Sampel dibuat dgn konteks di-restore per sampel (fix T0.2) + sampler ancestral
(32x). Prompt = 48B dari awal dokumen (setelah \\n\\n) di daerah unseen;
referensi = cuplikan riil daerah lain. Ambang awal dikalibrasi dari set riil.

Jalan (~2-3 mnt):
  python scripts/gen_health.py --ckpt checkpoints/gate_v2_tb26_186m.npz \\
      --corpus data/indonesian_big.bin --train_end 186000000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()


def lcs_len(a, b):
    """Panjang LCS (DP baris-tunggal, byte)."""
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for cb in b:
        cur = [0]
        ap = 0
        for i, ca in enumerate(a, 1):
            v = prev[i - 1] + 1 if ca == cb else (prev[i] if prev[i] > cur[-1] else cur[-1])
            cur.append(v)
        prev = cur
    return prev[len(a)]


def self_repeat_32(s):
    n = len(s)
    if n < 64:
        return 0.0
    seen, hit = set(), 0
    tot = n - 32 + 1
    for i in range(tot):
        w = bytes(s[i:i + 32])
        if w in seen:
            hit += 1
        else:
            seen.add(w)
    return hit / tot


def mem24_frac(s, train):
    """Fraksi jendela-24 yang muncul di korpus latih (bytes.find, C-speed)."""
    t = bytes(train)
    n = len(s) - 24 + 1
    if n <= 0:
        return 0.0
    step = max(1, n // 12)
    hit = tot = 0
    for i in range(0, n, step):
        tot += 1
        if t.find(bytes(s[i:i + 24])) >= 0:
            hit += 1
    return hit / tot


def longest_copy(s, train):
    t = bytes(train)
    best, n = 0, len(s)
    for i in range(0, n, 8):
        lo, hi = 24, min(200, n - i)
        if hi <= 24 or t.find(bytes(s[i:i + 24])) < 0:
            continue
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if t.find(bytes(s[i:i + mid])) >= 0:
                lo = mid
            else:
                hi = mid - 1
        best = max(best, lo)
    return best


def distinct4(s):
    n = len(s) - 4 + 1
    if n <= 0:
        return 0.0
    return len({bytes(s[i:i + 4]) for i in range(n)}) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gate_v2_tb26_186m.npz")
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--train_end", type=int, default=186_000_000)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--temps", default="0.7,1.0")
    ap.add_argument("--minps", default="0.0")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--jsonl", default="results/gen_health.jsonl")
    a = ap.parse_args()
    temps = [float(x) for x in a.temps.split(",")]
    minps = [float(x) for x in a.minps.split(",")]

    m, meta = StatCore.load(a.ckpt)
    print(f"ckpt: {a.ckpt} (latih {m._train_fed}B)", flush=True)
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    train = raw[:a.train_end]
    # prompt: 48B dari awal dokumen (setelah \n\n) di daerah unseen
    prompts, pos = [], a.train_end
    while len(prompts) < a.prompts and pos < len(raw) - 600:
        j = bytes(raw[pos:pos + 600]).find(b"\n\n")
        if j < 0:
            pos += 600
            continue
        prompts.append(bytes(raw[pos + j + 2:pos + j + 50]))
        pos += j + 600
    refs = [bytes(raw[a.train_end + 4_000_000 + i * 500:
                      a.train_end + 4_000_000 + i * 500 + 500])
            for i in range(a.prompts)]
    seeds = list(range(a.seeds))

    # kosakata held-out utk M5 (kata freq>=3 di daerah uji; dulu bug: region di
    # luar file -> vocab kosong -> M5=1 untuk semua). Pakai 2MB pertama daerah uji.
    vocab = set()
    vh = raw[a.train_end:a.train_end + 2_000_000]
    if len(vh) > 100000:
        txt = bytes(vh).decode("utf-8", errors="ignore").split()
        from collections import Counter
        c = Counter(txt)
        vocab = {w for w, k in c.items() if k >= 3}

    t0 = time.perf_counter()
    outs = {}
    for temp in temps:
        for mp in minps:
            key = f"T{temp}_m{mp}"
            outs[key] = []
            # G-0: seed per-prompt (1000+i*17+s); seed sama lintas-prompt = coupling
            # CRN yang menggelembungkan M1 (A=150 penuh -> B=62). Jangan kembalikan.
            for i, p in enumerate(prompts):
                for s in seeds:
                    g = m.generate(p, a.n, temp=temp, seed=1000 + i * 17 + s,
                                   ancestral=(mp == 0.0), min_p=mp)
                    outs[key].append(g)
    print(f"generation: {sum(len(v) for v in outs.values())} sampel "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    import itertools
    res = {}
    for key, samps in outs.items():
        tails = [s[100:350] for s in samps]  # potong 250B: LCS O(n^2) Python
        pairs = list(itertools.combinations(range(len(tails)), 2))[:300]
        m1 = max([lcs_len(tails[i], tails[j]) for i, j in pairs] + [0])
        # M2: seed berbeda, prompt sama -> samps diurut (prompt, seed)
        m2 = 0
        per = len(seeds)
        for pi in range(len(prompts)):
            grp = [s[len(prompts[pi]) + 64:len(prompts[pi]) + 314]
                   for s in samps[pi * per:(pi + 1) * per]]
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    m2 = max(m2, lcs_len(grp[i], grp[j]))
        m3 = float(np.mean([self_repeat_32(s) for s in samps]))
        m4 = float(np.mean([mem24_frac(s, train) for s in samps]))
        lc = int(np.mean([longest_copy(s, train)
                          for s in samps[:4]]))  # hanya 4 sampel (find mahal)
        toks = [w for s in samps
                for w in s.decode("utf-8", errors="ignore").split()]
        m5 = 1.0 - (sum(1 for w in toks if w in vocab) / max(len(toks), 1))
        # M6: kalibrasi — skor sampel oleh model beku yang sama
        ces = []
        for s in samps[:8]:
            ces.append(float(np.nanmean(m.eval_stream_fast(np.frombuffer(s, dtype=np.uint8)))))
        h = float(np.nanmean(m.eval_stream_fast(
            np.frombuffer(bytes().join(refs)[:20000], dtype=np.uint8))))
        m6 = abs(float(np.mean(ces)) - h)
        m7 = float(np.mean([distinct4(s) for s in samps]))
        res[key] = dict(M1=m1, M2=m2, M3=round(m3, 4), M4=round(m4, 4),
                        longest_copy=lc, M5=round(m5, 4), M6=round(m6, 4),
                        M7=round(m7, 4))
        print(f"[{key}] M1={m1} M2={m2} M3={m3:.4f} M4={m4:.4f} copy={lc} "
              f"M5={m5:.4f} M6={m6:.4f} M7={m7:.4f}", flush=True)
    # baseline riil utk kalibrasi ambang
    rr = dict(M1=max(lcs_len(refs[i][100:], refs[j][100:])
                     for i in range(len(refs)) for j in range(i + 1, len(refs))),
              M3=round(float(np.mean([self_repeat_32(r) for r in refs])), 4),
              M4=round(float(np.mean([mem24_frac(r, train) for r in refs])), 4),
              M5=round(1.0 - sum(1 for r in refs for w in r.decode(
                  "utf-8", errors="ignore").split() if w in vocab) / max(
                      sum(len(r.decode("utf-8", errors="ignore").split())
                          for r in refs), 1), 4),
              M7=round(float(np.mean([distinct4(r) for r in refs])), 4))
    print(f"[RIIL] {rr}", flush=True)
    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ckpt=a.ckpt, res=res, riil=rr)) + "\n")


if __name__ == "__main__":
    main()
