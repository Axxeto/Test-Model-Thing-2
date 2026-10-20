"""Baseline yang ADIL: n-gram di atas token BPE yang sama, tokenizer dilatih HANYA pada prefiks.

Konteks: klaim lama "TMT2 mengalahkan n-gram order 2" membandingkan model level TOKEN dengan
n-gram level BYTE. N-gram byte hanya punya 2 byte konteks, sedangkan n-gram token punya ~7 byte
konteks -- jadi baseline byte itu lebih lemah. Skrip ini menghitung baseline di level token yang
sama, dan melatih tokenizer hanya pada prefiks agar tidak ada kebocoran ke bagian "belum terlihat".

  python scripts/fair_baseline.py --n 1121681 --vocab 4096 --tok_frac 0.2
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.corpus import load
from tmt2 import tokenizer as tk


def ngram_bits_tokens(ids, order, V):
    """BPB prekuensial n-gram adaptif (back-off) atas id token; sparse agar hemat memori."""
    counts = [dict() for _ in range(order + 1)]
    out = np.empty(len(ids))
    for i, b in enumerate(ids):
        p = 1.0 / V
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o].get(tuple(ids[i - o:i]))
            if c is None:
                c = counts[o][tuple(ids[i - o:i])] = [{}, 0.0]
            p = (c[0].get(b, 0.0) + 2.0 * p) / (c[1] + 2.0)
        out[i] = -np.log2(p)
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o][tuple(ids[i - o:i])]
            c[0][b] = c[0].get(b, 0.0) + 1
            c[1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--n", type=int, default=1121681)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tok_frac", type=float, default=0.2, help="porsi korpus untuk melatih tokenizer")
    ap.add_argument("--orders", default="0,1,2")
    a = ap.parse_args()

    raw = load(a.data)[:a.n]
    cut = int(len(raw) * a.tok_frac)
    tok = tk.train(raw[:cut], a.vocab, None)          # HANYA prefiks -> tanpa kebocoran
    ids_arr, lens, canon = tk.encode(tok, raw)
    ids = [int(x) for x in ids_arr]
    cum = np.cumsum(lens)
    i_cut = int(np.searchsorted(cum, cut, side="right"))

    print(f"korpus   : {len(canon)} byte -> {len(ids)} token ({len(canon)/len(ids):.2f} byte/token)")
    print(f"tokenizer: dilatih pada {cut} byte pertama saja ({a.tok_frac*100:.0f}% korpus)")
    print(f"batas    : byte {cut} = token {i_cut}")
    print()
    print(f"{'model':<20}{'bpb semua':>11}{'bpb setelah prefiks':>22}")
    for o in [int(x) for x in a.orders.split(",")]:
        bits = ngram_bits_tokens(ids, o, a.vocab)[1:]     # token pertama tanpa prediksi
        bl = lens[1:]
        print(f"token {o}-gram{'':<9}{bits.sum()/bl.sum():>11.3f}"
              f"{bits[i_cut:].sum()/bl[i_cut:].sum():>22.3f}")
    print()
    print("Bandingkan dengan TMT2 pada kolom 'setelah prefiks' (jalankan bigtest dengan")
    print("--bpe --tok_frac yang sama), karena kolom itu bebas kebocoran tokenizer.")
    print("Catatan: smoothing (c+2p)/(n+2) terlalu lemah untuk konteks jarang sehingga")
    print("trigram bisa tampak lebih buruk dari bigram; baseline ini masih LONGGAR.")


if __name__ == "__main__":
    main()
