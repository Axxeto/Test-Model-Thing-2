"""BASELINE BPE ADIL pada SPLIT YANG SAMA — melengkapi Gate 1.

Protokol: tokenizer dilatih HANYA pada data[:ntrain] (tanpa kebocoran), lalu n-gram token
dibekukan setelah fase latih dan dinilai pada data[ntrain:ntrain+ntest]. Hasilnya dikonversi
ke bit/byte supaya sebanding dengan BPB TMT2.

Ini baseline TERKUAT untuk pembanding level-token: token punya konteks efektif ~3-4 byte
per token, jadi n-gram token orde-2 sudah memakai ~8 byte konteks.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2 import tokenizer as tk


def bits_frozen(train_ids, eval_ids, order, V):
    counts = [dict() for _ in range(order + 1)]
    for i, b in enumerate(train_ids):
        for o in range(order + 1):
            if i < o:
                break
            key = tuple(train_ids[i - o:i])
            c = counts[o].get(key)
            if c is None:
                c = counts[o][key] = [{}, 0.0]
            c[0][b] = c[0].get(b, 0.0) + 1.0
            c[1] += 1.0
    hist = list(train_ids)
    out = np.zeros(len(eval_ids))
    base = len(train_ids)
    for j, b in enumerate(eval_ids):
        p = 1.0 / V
        i = base + j
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o].get(tuple(hist[-o:]) if o else ())
            if c is None:
                continue
            p = (c[0].get(b, 0.0) + 2.0 * p) / (c[1] + 2.0)
        out[j] = -np.log2(p)
        hist.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--orders", default="0,1,2")
    a = ap.parse_args()

    raw = load(a.data)
    tr = bytes(raw[:a.ntrain])
    te = bytes(raw[a.ntrain:a.ntrain + a.ntest])
    tok = tk.train(tr, a.vocab, None)                 # HANYA bagian latih -> tanpa kebocoran
    ids_tr, lens_tr, _ = tk.encode(tok, tr)
    ids_te, lens_te, _ = tk.encode(tok, te)
    ids_tr = [int(x) for x in ids_tr]
    ids_te = [int(x) for x in ids_te]
    b_tr = int(np.sum(lens_tr))
    b_te = int(np.sum(lens_te))
    print(f"data={a.data} ntrain={a.ntrain} ntest={a.ntest} vocab={a.vocab}")
    print(f"token latih={len(ids_tr)} ({b_tr/len(ids_tr):.2f} B/tok) | "
          f"token uji={len(ids_te)} ({b_te/len(ids_te):.2f} B/tok)")

    print(f"\n{'model':<22}{'bit/token':>12}{'bit/byte':>12}")
    print("-" * 46)
    for o in [int(x) for x in a.orders.split(",")]:
        bits = bits_frozen(ids_tr, ids_te, o, a.vocab)
        print(f"token {o}-gram{'':<10}{bits.sum()/len(ids_te):>12.4f}"
              f"{bits.sum()/b_te:>12.4f}", flush=True)
    print("\nPembanding: TMT2+mixer pada split yang sama = 1,8273 bit/byte.")


if __name__ == "__main__":
    main()
