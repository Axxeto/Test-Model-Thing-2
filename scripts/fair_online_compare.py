"""PERBANDINGAN ADIL (held-out): TMT2 vs baseline konteks-lokal, protokol SAMA.

Protokol: semua model "dilatih" pada data[:ntrain] dan dinilai pada data[ntrain:ntrain+ntest]
YANG TIDAK PERNAH DILIHAT. Baseline n-gram/ctx-mix: tabel dibekukan setelah fase latih.
TMT2: bobot dibekukan (train=False), state dibawa (warm) DAN dingin (reset) dilaporkan.

Ini gate yang harus dilewati sebelum klaim "model bahasa berguna".
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


def ctxmix_frozen(tr, te, order=4, D=0.75, V=256):
    """ctx-mix adaptif (absolute discounting) DIBEKUKAN setelah `tr`; CE pada `te`.

    CATATAN: `scripts/strong_baseline.py:ctx_mix_bpb` memakai add-alpha `(c + alpha*p)/(tot+alpha)`
    dengan alpha=0,5 TETAP. Untuk konteks dengan `tot` besar itu memusnahkan massa backoff
    (p_ctx ~ alpha*p/tot). Dampaknya bergantung protokol: ONLINE hanya ~0,02 bit terlalu tinggi
    (ctx-mix-3 2,4095 -> 2,3870), tetapi pada jendela TAHAN dengan tabel beku CE meledak ke
    17,3 bit. Versi di sini memakai diskon absolut (massa backoff = D * jumlah penerus berbeda),
    dan JUGA memperbaiki jebakan protokol: `hist` (konteks) HARUS tetap berjalan saat eval —
    hanya COUNTS yang dibekukan. Kalau konteks ikut dibekukan, CE palsu 12-17 bit.
    """
    counts = [dict() for _ in range(order + 1)]
    tot = [dict() for _ in range(order + 1)]
    uniq = [dict() for _ in range(order + 1)]
    uni = np.zeros(V)
    hist = bytearray()

    def update(b):
        uni[b] += 1
        for o in range(1, order + 1):
            if len(hist) < o:
                break
            key = bytes(hist[-o:])
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

    def predict(b):
        p = (uni + 1e-9) / (uni.sum() + 1e-9 * V)          # (V,)
        for o in range(1, order + 1):
            if len(hist) < o:
                break
            key = bytes(hist[-o:])
            c = counts[o].get(key)
            if c is None:
                continue
            n, t = uniq[o][key], tot[o][key]
            mass = D * n if t > 0 else 0.0
            q = (mass / t) * p
            for w, cw in c.items():
                q[w] += max(cw - D, 0.0) / t
            p = q
        return -np.log2(p[b] + 1e-12)

    for b in tr:
        update(int(b))
        hist.append(int(b))
    # EVAL BEKU: tabel TIDAK diperbarui, tetapi KONTEKS tetap berjalan (hist bertambah).
    # Tanpa ini konteks menjadi basi dan CE meledak (terukur 12-17 bit) — bukan baseline yang sah.
    ce = 0.0
    for b in te:
        b = int(b)
        ce += predict(b)
        hist.append(b)
    return ce / len(te)


def jm_ngram(tr, te, order=5, lam=0.5, V=256):
    """N-gram Jelinek-Mercer dibekukan setelah `tr`; CE pada `te`."""
    tabs, tots = [], []
    for o in range(1, order + 1):
        tab, tot = {}, {}
        for i in range(o, len(tr)):
            c = tuple(int(x) for x in tr[i - o:i])
            b = int(tr[i])
            tab.setdefault(c, np.zeros(V))[b] += 1.0
            tot[c] = tot.get(c, 0.0) + 1.0
        tabs.append(tab)
        tots.append(tot)
    uni = np.bincount(np.asarray(tr), minlength=V).astype(float)
    uni = (uni + 1e-9) / (uni.sum() + 1e-9 * V)
    ce = 0.0
    for i in range(1, len(te)):
        ctx = tuple(int(x) for x in te[max(0, i - order):i])
        b = int(te[i])
        p = uni
        for o in range(1, order + 1):
            c = ctx[-o:]
            if len(c) < o:
                continue
            cc = tabs[o - 1].get(c)
            if cc is None:
                continue
            p = lam * (cc / tots[o - 1][c]) + (1 - lam) * p
        ce += -np.log2(p[b] + 1e-12)
    return ce / (len(te) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)} d={a.d} L={a.L} seed={a.seed}")
    print(f"{'metode':<30}{'bit/byte':>10}")
    print("-" * 42)

    t0 = time.time()
    print(f"{'ctx-mix orde-4 (beku, diskon)':<30}{ctxmix_frozen(tr, te):>10.4f}   "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    print(f"{'n-gram JM orde-5 (beku)':<30}{jm_ngram(tr, te):>10.4f}   ({time.time()-t0:.0f}s)",
          flush=True)

    cfg = Config.best(d=a.d, n_layers=a.L, vocab=256, d_mem=max(8, a.d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, think_max=0,
                      use_innovation=False, seg_len=10 ** 9, seed=a.seed)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, seed=a.seed, use_stop=False)
    L = FastLearner(m, o)
    s = Stream(m, online=True)
    t0 = time.time()
    for b in tr:
        L.step(s, int(b), train=True)
    # (a) WARME: lanjutkan aliran yang SAMA, bobot beku -> state membawa konteks
    ces = []
    for b in te:
        c = L.step(s, int(b), train=False)
        if c is not None:
            ces.append(c)
    warm = float(np.mean(ces) / LN2)
    # (b) DINGIN: state direset
    cold = float(L.eval_stream([int(b) for b in te]).mean() / LN2)
    print(f"{'TMT2+mixer (state HANGAT)':<30}{warm:>10.4f}   ({time.time()-t0:.0f}s)")
    print(f"{'TMT2+mixer (state DINGIN)':<30}{cold:>10.4f}")

    # Referensi kompresor. Dua protokol, keduanya dilaporkan agar tidak menyesatkan:
    #  (a) jendela uji SENDIRI  -> terlalu keras untuk kompresor (jendela pendek, tanpa konteks)
    #  (b) KONDISIONAL: bit tambahan untuk mengkode `tr+te` setelah `tr` sudah dikode.
    #      Ini "bit/byte untuk te DENGAN tr sebagai konteks" = sebanding dengan BPB held-out TMT2.
    import bz2, lzma, zlib
    def cond(comp):
        full = comp(bytes(tr) + bytes(te))
        base = comp(bytes(tr))
        return 8.0 * (len(full) - len(base)) / len(te)
    print(f"\n{'bz2 -9 (jendela uji saja)':<30}{8*len(bz2.compress(te,9))/len(te):>10.4f}")
    print(f"{'lzma -9 (jendela uji saja)':<30}"
          f"{8*len(lzma.compress(te, preset=9))/len(te):>10.4f}")
    print(f"\n{'bz2 -9 KONDISIONAL (tr|te)':<30}{cond(lambda b: bz2.compress(b, 9)):>10.4f}")
    print(f"{'lzma -9 KONDISIONAL (tr|te)':<30}"
          f"{cond(lambda b: lzma.compress(b, preset=9)):>10.4f}")
    print(f"{'zlib -9 KONDISIONAL (tr|te)':<30}{cond(lambda b: zlib.compress(b, 9)):>10.4f}")


if __name__ == "__main__":
    main()
