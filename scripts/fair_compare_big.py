"""Perbandingan ADIL Tingkat A vs baseline pada korpus Indonesia besar.

ATURAN (jebakan protokol #4/#11 di DOKUMENTASI_LENGKAP.md):
  * Jendela uji = EKOR aliran, SAMA persis dengan train_big_id.py (``ntest`` byte terakhir).
    Baseline TIDAK boleh diukur di awal korpus.
  * Dua mode baseline, keduanya dilaporkan:
      - "beku" : kompresor hanya "melihat" data latih, lalu menekan jendela uji
                 (out-of-sample; biaya byte uji = len(tr+te) - len(tr)).
      - "lihat": kompresor menekan jendela uji itu sendiri (batas atas tidak adil).
    ctx-mix / match = model ONLINE (prekuensial) -> dijalankan pada tr_sub+te sebagai SATU
    aliran, lalu CE dirata-rata HANYA pada bagian uji (bukan seluruh array).
  * Biaya bz2/lzma O(n) pada 190 MB terlalu berat; mode "beku" memakai SUBSET LATIH
    (--train_subset) dan itu dilaporkan apa adanya.

Jalan:
  python scripts/fair_compare_big.py --ntest 100000 --train_subset 8000000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

import bz2
import lzma
import zlib

LN2 = float(np.log(2))


# ------------------------------------------------------------------ baseline online (CE per-byte)
def ctx_mix_ce_perbyte(raw, order=4, D=0.75, V=256):
    """Sama seperti strong_baseline.ctx_mix_bpb tetapi mengembalikan ARRAY CE per-byte
    (bit) supaya bisa diiris ke jendela uji.

    Catatan: ``strong_baseline`` memanggil versinya dengan ``bytes`` (kunci = potongan bytes,
    hashable). Di sini kita terima ``ndarray`` untuk kenyamanan, jadi konteks dijadikan
    ``bytes`` eksplisit (``raw[i-o:i].tobytes()``).
    """
    raw = np.asarray(raw, dtype=np.uint8).tobytes()
    counts = [dict() for _ in range(order + 1)]
    tot = [dict() for _ in range(order + 1)]
    uniq = [dict() for _ in range(order + 1)]
    uni = np.zeros(V)
    out = np.empty(len(raw), dtype=np.float64)
    for i, b in enumerate(raw):
        b = int(b)
        p = (uni + 1e-9) / (uni.sum() + 1e-9 * V)
        for o in range(1, order + 1):
            if i < o:
                break
            key = raw[i - o:i]
            c = counts[o].get(key)
            if c is None:
                continue
            t = tot[o][key]
            q = (D * uniq[o][key] / t) * p
            for w, cw in c.items():
                q[w] += max(cw - D, 0.0) / t
            p = q
        out[i] = -np.log2(p[b] + 1e-12)
        uni[b] += 1
        for o in range(1, order + 1):
            if i < o:
                break
            key = raw[i - o:i]
            c = counts[o].get(key)
            if c is None:
                c = counts[o][key] = {}
                tot[o][key] = 0.0
                uniq[o][key] = 0
            if c.get(b) is None:
                c[b] = 0.0
                uniq[o][key] += 1
            c[b] += 1.0
            tot[o][key] += 1.0
    return out


def match_ce_perbyte(raw, maxlen=8, alpha=0.5, V=256, min_ctx=1):
    """Versi array-per-byte dari match_model_bpb (bug kebocoran sudah ditutup di sana)."""
    raw = np.asarray(raw, dtype=np.uint8).tobytes()
    last_pos = {}
    out = np.empty(len(raw), dtype=np.float64)
    n = len(raw)
    uni = np.zeros(V)
    for i in range(n):
        b = int(raw[i])
        p = (uni + alpha) / (uni.sum() + alpha * V)
        best = None
        for L in range(min(maxlen, i), min_ctx - 1, -1):
            key = raw[i - L:i]
            d = last_pos.get(L)
            if d is None:
                continue
            j0 = d.get(key)
            if j0 is None or j0 + L >= i:
                continue
            best = int(raw[j0 + L])
            break
        if best is not None:
            q = p.copy()
            q[best] += 1.0
            p = q / q.sum()
        out[i] = -np.log2(p[b] + 1e-12)
        uni[b] += 1
        for L in range(min_ctx, min(maxlen, i) + 1):
            j0 = i - L
            last_pos.setdefault(L, {})[raw[j0:i]] = j0
    return out


# ------------------------------------------------------------------ kompresor (beku / lihat)
def comp_len(comp_fn, data):
    return len(comp_fn(bytes(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntest", type=int, default=100000)
    ap.add_argument("--train_subset", type=int, default=8000000,
                    help="byte data latih untuk baseline 'beku' (bz2/lzma mahal O(n))")
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--level", type=int, default=9)
    ap.add_argument("--jsonl", default="results/fair_compare_big.jsonl")
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    total = raw.size
    te = raw[total - a.ntest:]
    tr = raw[:min(a.train_subset, total - a.ntest)]
    stream = np.concatenate([tr, te])
    split = len(tr)
    print(f"korpus {total} B | uji(ekor) {len(te)} B | subset latih baseline {len(tr)} B")

    maps = {"bz2": lambda r: bz2.compress(r, a.level),
            "lzma": lambda r: lzma.compress(r, preset=a.level),
            "zlib": lambda r: zlib.compress(r, a.level)}

    rows = []
    print(f"\n{'metode':<24}{'beku':>10}{'lihat':>10}{'detik':>9}")
    print("-" * 53)
    for nm, fn in maps.items():
        t0 = time.time()
        base = comp_len(fn, tr)
        both = comp_len(fn, stream)
        beku = (both - base) * 8.0 / max(len(te), 1)
        lihat = comp_len(fn, te) * 8.0 / max(len(te), 1)
        dt = time.time() - t0
        print(f"{nm + ' -' + str(a.level):<24}{beku:>10.4f}{lihat:>10.4f}{dt:>9.1f}")
        rows.append(dict(corpus=a.corpus, ntest=len(te), train_subset=len(tr),
                         method=f"{nm}-{a.level}", mode_beku=beku, mode_lihat=lihat))

    for o in [int(x) for x in a.orders.split(",")]:
        t0 = time.time()
        ce = ctx_mix_ce_perbyte(stream, order=o)
        v = float(ce[split:].mean())            # HANYA bagian uji
        dt = time.time() - t0
        print(f"{'ctx-mix orde ' + str(o):<24}{v:>10.4f}{v:>10.4f}{dt:>9.1f}")
        rows.append(dict(corpus=a.corpus, ntest=len(te), train_subset=len(tr),
                         method=f"ctx-mix-{o}", mode_beku=v, mode_lihat=v, online=True))

    t0 = time.time()
    ce = match_ce_perbyte(stream)
    v = float(ce[split:].mean())
    dt = time.time() - t0
    print(f"{'match model':<24}{v:>10.4f}{v:>10.4f}{dt:>9.1f}")
    rows.append(dict(corpus=a.corpus, ntest=len(te), train_subset=len(tr),
                     method="match", mode_beku=v, mode_lihat=v, online=True))

    best = min(rows, key=lambda r: r["mode_beku"])
    print(f"\nbaseline terbaik (beku): {best['method']} = {best['mode_beku']:.4f} bit/byte")
    os.makedirs(os.path.dirname(a.jsonl) or ".", exist_ok=True)
    with open(a.jsonl, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"ditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
