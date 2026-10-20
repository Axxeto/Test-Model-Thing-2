"""BASELINE KUAT pada byte yang sama: lzma, bz2, PPM sederhana, dan match model.

Tujuan: menetapkan TARGET yang harus dikalahkan TMT2, pada korpus & protokol yang sama
(bit/byte prekuensial). Murah dan menentukan arah.

Catatan satuan: kompresor menghasilkan BYTE keluaran; ``bit/byte`` = 8 * len(compressed) / len(raw).
Untuk PPM/match kita hitung CE prekuensial langsung (bit/byte), apples-to-apples dengan TMT2.
"""
import argparse, bz2, json, lzma, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.corpus import load, sanity_check

LN2 = float(np.log(2))


def comp_bpb(raw, comp):
    return 8.0 * len(comp) / len(raw)


def lzma_bpb(raw, preset=9):
    return comp_bpb(raw, lzma.compress(raw, preset=preset))


def bz2_bpb(raw, level=9):
    return comp_bpb(raw, bz2.compress(raw, compresslevel=level))


def zlib_bpb(raw, level=9):
    import zlib
    return comp_bpb(raw, zlib.compress(raw, level))


def ctx_mix_bpb(raw, order=4, D=0.75, V=256):
    """PPM-lite: interpolasi orde 0..order dengan back-off, CE prekuensial. Mengembalikan bit/byte.

    ⚠️ DIPERBAIKI 2026-10-04. Versi lama memakai add-alpha TETAP ``(c + alpha*p)/(tot+alpha)``
    dengan alpha=0,5. Untuk konteks dengan ``tot`` besar itu memusnahkan massa back-off
    (p_ctx ~ alpha*p/tot). Besarnya dampak BERGANTUNG PROTOKOL:
      * ONLINE (fungsi ini, konteks selalu terlihat): hanya ~0,02 bit terlalu tinggi
        (ctx-mix-3 lama 2,4095 vs baru 2,3870) — angka lama masih kira-kira sah.
      * BEKU (tabel dibekukan, jendela tahan): CE MELEDAK ke 17,3 bit — sama sekali tidak sah.
    Sekarang memakai **diskon absolut**: massa back-off = D x (jumlah penerus berbeda).
    Versi terkontrol (beku + konteks berjalan) ada di
    ``scripts/fair_online_compare.py:ctxmix_frozen``.
    """
    counts = [dict() for _ in range(order + 1)]
    tot = [dict() for _ in range(order + 1)]
    uniq = [dict() for _ in range(order + 1)]
    uni = np.zeros(V)
    ce = 0.0
    n = 0
    for i, b in enumerate(raw):
        b = int(b)
        # prediksi
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
        ce += -np.log2(p[b] + 1e-12)
        n += 1
        # update
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
    return ce / n


def match_model_bpb(raw, maxlen=8, alpha=0.5, V=256, min_ctx=1):
    """Match model prekuensial: cari kecocokan terpanjang dengan riwayat SEBELUM posisi ini.

    PERINGATAN (bug yang sudah diperbaiki). Implementasi pertama menyimpan posisi
    ``key = raw[i-L:i]`` -> ``i`` SETELAH memprediksi, sehingga kecocokan terakhir untuk
    konteks itu hampir selalu ``j = i-1`` yang penerusnya ``raw[j+1] = raw[i]`` = byte
    yang sedang diprediksi -> **kebocoran total** (akurasi 1.0, CE ~0). Perbaikan:
      * konteks disimpan sebagai ``raw[i-L:i]`` -> posisi AWAL ``j0`` dari kemunculan
        sebelumnya, dan penerusnya ``raw[j0+L]``;
      * kemunculan yang menutupi posisi sekarang (``j0 + L >= i``) DITOLAK.
    """
    last_pos = {}                       # L -> {key: j0} (posisi awal kemunculan)
    ce = 0.0
    n = len(raw)
    uni = np.zeros(V)
    for i in range(n):
        b = int(raw[i])
        p = (uni + alpha) / (uni.sum() + alpha * V)
        best, bestL = None, 0
        for L in range(min(maxlen, i), min_ctx - 1, -1):
            key = raw[i - L:i]
            d = last_pos.get(L)
            if d is None:
                continue
            j0 = d.get(key)
            if j0 is None:
                continue
            # tolak kemunculan yang menutupi posisi sekarang (penerus tak teramati / bocor)
            if j0 + L >= i:
                continue
            best, bestL = int(raw[j0 + L]), L
            break
        if best is not None:
            q = p.copy()
            q[best] += 1.0
            p = q / q.sum()
        ce += -np.log2(p[b] + 1e-12)
        uni[b] += 1
        for L in range(min_ctx, min(maxlen, i) + 1):
            j0 = i - L
            last_pos.setdefault(L, {})[raw[j0:i]] = j0
    return ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="wikitext")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--jsonl", default="results/strong_baseline.jsonl")
    a = ap.parse_args()

    raw = load(a.kind, n=a.n)
    sanity_check(raw, name=f"baseline:{a.kind}", expect_len=a.n)
    print(f"\nkorpus={a.kind} n={len(raw)} byte\n")
    rows = []
    print(f"{'metode':<22}{'bit/byte':>10}{'detik':>8}")
    print("-" * 40)
    for name, fn in [("lzma -9", lambda r: lzma_bpb(r, 9)),
                     ("bz2 -9", lambda r: bz2_bpb(r, 9)),
                     ("zlib -9", lambda r: zlib_bpb(r, 9))]:
        t0 = time.time()
        v = fn(raw)
        print(f"{name:<22}{v:>10.4f}{time.time()-t0:>8.1f}")
        rows.append(dict(kind=a.kind, n=len(raw), method=name, bpb=v))
    for o in [int(x) for x in a.orders.split(",")]:
        t0 = time.time()
        v = ctx_mix_bpb(raw, order=o)
        print(f"{'ctx-mix orde '+str(o):<22}{v:>10.4f}{time.time()-t0:>8.1f}")
        rows.append(dict(kind=a.kind, n=len(raw), method=f"ctx-mix-{o}", bpb=v))
    t0 = time.time()
    v = match_model_bpb(raw)
    print(f"{'match model':<22}{v:>10.4f}{time.time()-t0:>8.1f}")
    rows.append(dict(kind=a.kind, n=len(raw), method="match", bpb=v))

    best = min(rows, key=lambda r: r["bpb"])
    print(f"\nTARGET TERBAIK: {best['method']} = {best['bpb']:.4f} bit/byte")
    os.makedirs(os.path.dirname(a.jsonl) or ".", exist_ok=True)
    with open(a.jsonl, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
