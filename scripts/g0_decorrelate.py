"""G-0 — dekorelasi M1: CRN-coupling vs basin nyata (tanpa kode produksi, ~10 mnt).

Kondisi (n=250, T=1.0, ancestral, ckpt 186M V2):
  A: seed=0 semua prompt, ctx apa-adanya (protokol lama gen_health.py).
  B: seed=1000+i per prompt (putus coupling uniform).
  C: seed per-prompt + konteks DINGIN (hist/pos dinolkan; putative corpus-tail).
  D: seed per-prompt + prefiks pemisah "\\n\\n".

Metrik per kondisi: M1 (LCS-silang, ekor 250B), M2, indeks koalesensi
(posisi pertama 24B-terakhir identik antar-pasang), surprisal tersampel
(CE sampel di bawah model beku — tanpa akumulator kode, via eval_stream_fast).

Aturan putus: M1(B,C)<=100 & M1(A)~250 -> artefak CRN | M1(B,C)>=200 &
surprisal<1,2 -> basin | M1>=200 & surprisal~2 -> memori-pendek struktural.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()


def lcs_len(a, b):
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for cb in b:
        cur = [0]
        for i, ca in enumerate(a, 1):
            cur.append(prev[i - 1] + 1 if ca == cb
                       else (prev[i] if prev[i] > cur[-1] else cur[-1]))
        prev = cur
    return prev[len(a)]


def coalesce_idx(a, b, w=24):
    la, lb = len(a), len(b)
    for i in range(min(la, lb) - w + 1):
        if a[i:i + w] == b[i:i + w]:
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gate_v2_tb26_186m.npz")
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--train_end", type=int, default=186_000_000)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--jsonl", default="results/g0_decorrelate.jsonl")
    a = ap.parse_args()

    m, _ = StatCore.load(a.ckpt)
    base_ctx = m.stream_ctx()
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    prompts, pos = [], a.train_end
    while len(prompts) < 8 and pos < len(raw) - 600:
        j = bytes(raw[pos:pos + 600]).find(b"\n\n")
        if j < 0:
            pos += 600
            continue
        prompts.append(bytes(raw[pos + j + 2:pos + j + 50]))
        pos += j + 600

    out = {}
    for cond in ("A", "B", "C", "D"):
        samps = []
        for i, p in enumerate(prompts):
            seed = 0 if cond == "A" else 1000 + i
            if cond == "C":
                m.hist[:] = 0
                m.pos, m.hlen, m.prev_byte = 0, 0, 0
            pre = b"\n\n" if cond == "D" else b""
            # generate() restore ctx sendiri (T0.2); C: restore manual setelahnya
            g = m.generate(pre + p, a.n, temp=1.0, seed=seed, ancestral=True)
            samps.append(g)
            if cond == "C":
                m.restore_ctx(base_ctx)
        m.restore_ctx(base_ctx)
        tails = [s[100:350] if len(s) > 350 else s[100:] for s in samps]
        pairs = list(itertools.combinations(range(len(tails)), 2))[:100]
        m1 = max([lcs_len(tails[i], tails[j]) for i, j in pairs] + [0])
        coal = [coalesce_idx(tails[i], tails[j])
                for i, j in pairs[:40]]
        coal = [c for c in coal if c >= 0]
        surp = float(np.mean([
            float(np.nanmean(m.eval_stream_fast(
                np.frombuffer(s, dtype=np.uint8)))) for s in samps[:8]]))
        out[cond] = dict(M1=m1,
                         coal_med=(float(np.median(coal)) if coal else -1.0),
                         coal_frac=round(len(coal) / 40, 2),
                         surprisal=round(surp, 4))
        print(f"[{cond}] M1={m1} koalesensi_med={out[cond]['coal_med']} "
              f"frac={out[cond]['coal_frac']} surprisal={surp:.4f}", flush=True)
    print("== PUTUSAN G-0 ==", flush=True)
    if out["A"]["M1"] >= 200 and out["B"]["M1"] <= 100:
        print("artefak CRN: perbaiki metrik (seed per-prompt), hentikan G3-G6", flush=True)
    elif out["B"]["M1"] >= 200 and out["B"]["surprisal"] < 1.2:
        print("basin berentropi rendah: kerjakan G3, lalu G4", flush=True)
    elif out["B"]["M1"] >= 200:
        print("memori-pendek struktural: hanya G6 relevan", flush=True)
    else:
        print("campuran/tak-putus: lihat angka", flush=True)
    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
