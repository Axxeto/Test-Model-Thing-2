"""M5 — INTEGRASI A+B: apakah Tingkat B menambah di atas Tingkat A?

Gerbang G2 (ARCHITECTURE_V2.md §7): "Tingkat B menambah >= +0,05 bit di atas Tingkat A".
Bila gagal -> keluarkan dari jalur produksi, simpan sebagai modul riset, tulis hasil negatif.

Alur:
  1. Latih Tingkat B (embed + blok Gamelan + kepala) pada byte CE, teacher-forced,
     BPTT di dalam deret (bentuk rekuren; gradiennya identik dengan bentuk chunkwise).
  2. Bekukan. Hitung `stretch(p(bit|prefiks))` untuk tiap bit tiap byte (pohon biner).
  3. Latih Tingkat A dengan dan tanpa masukan neural itu; bandingkan BPB tahan.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.gamelan.block import GamelanBlock
from tmt2.gamelan.train import block_forward_cache, block_backward
from tmt2.stat_core import StatCore, build_stretch
from tmt2.model import Config

LN2 = float(np.log(2))
CORPORA = {"wikitext": "wikitext", "indonesian": "data/indonesian_1m_pure.txt"}


class Adam:
    def __init__(self, params, lr):
        self.p = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)


def bit_probs(logits, bytes_):
    """(N,256) logits + byte sebenarnya -> (N,8) p(bit=1 | prefiks byte itu)."""
    lo = logits - logits.max(1, keepdims=True)
    P = np.exp(lo)
    P /= P.sum(1, keepdims=True)
    n = len(bytes_)
    out = np.zeros((n, 8))
    for i in range(8):
        pref = (bytes_ >> (8 - i)) & ((1 << i) - 1) if i else np.zeros(n, dtype=np.int64)
        lo_i = pref << (8 - i)
        half = 1 << (7 - i)
        idx = (lo_i[:, None] + np.arange(1 << (8 - i))[None, :])
        block = P[np.arange(n)[:, None], idx]
        out[:, i] = block[:, :half].sum(1) / (block.sum(1) + 1e-300)
    return out


def train_tierB(data, d, heads, conv_k, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1.0, (256, d))
    W_out = rng.normal(0, 1.0 / np.sqrt(d), (256, d))
    blk = GamelanBlock(d=d, heads=heads, conv_k=conv_k, seed=seed)
    params = {"E": E, "W_out": W_out}
    for nm in ("wd", "bd", "wp", "wq", "wk", "wv", "wl", "bl", "wo", "wg", "bg",
               "w1", "w2", "g1", "g2"):
        params[nm] = getattr(blk, nm)
    opt = Adam(params, lr)
    N = len(data)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for s in range(0, N - batch - 1, batch):
            b = data[s:s + batch].astype(np.int64)
            tgt = data[s + 1:s + batch + 1].astype(np.int64)
            xb = E[b]
            y, cache = block_forward_cache(blk, xb)
            logits = y @ W_out.T
            lo = logits - logits.max(1, keepdims=True)
            Ex = np.exp(lo)
            P = Ex / Ex.sum(1, keepdims=True)
            loss = float(-np.mean(np.log(P[np.arange(len(b)), tgt] + 1e-300)))
            tot += loss
            nb += 1
            dlog = P.copy()
            dlog[np.arange(len(b)), tgt] -= 1.0
            dlog /= len(b)
            dy = dlog @ W_out
            dx, g = block_backward(blk, cache, dy)
            grads = {k: v for k, v in g.items()}
            grads["W_out"] = dlog.T @ y
            xgrad = np.zeros_like(E)                 # gradien embedding dari dL/dx tiap posisi
            np.add.at(xgrad, b, dx)
            grads["E"] = xgrad
            opt.step(grads)
        print(f"    epoch {ep+1}: CE latih {tot/max(1,nb)/LN2:.4f} bit", flush=True)
    return blk, E, W_out


def nn_stretch(blk, E, W_out, data, chunk=128):
    """`stretch(p(bit|prefiks))` per byte, SELARAS dengan protokol prekuensial Tier A.

    Kunci keselarasan: ``out[i]`` = prediksi untuk byte **i**, dibuat dari state SETELAH byte i-1.
    Di dalam blok, ``y[t]`` (setelah melihat byte s+t) memprediksi byte s+t+1, jadi keluarannya
    harus diletakkan di indeks ``s+t+1`` — DAN prefiks yang dipakai untuk pohon biner harus
    prefiks byte yang DIPREDIKSI (s+t+1), bukan byte yang sudah dilihat.

    Versi pertama lupa keduanya -> off-by-one -> Tier A bisa membaca byte i dari masukan neural
    yang seharusnya belum melihatnya. Itu kebocoran, dan terbukti: Tingkat B ACAK pun "menang"
    +0,26 bit sebelum diperbaiki.
    """
    N = len(data)
    out = np.zeros((N, 8))
    for s in range(0, N - 1, 4096):
        e = min(s + 4096, N)
        b = data[s:e].astype(np.int64)
        y, _ = blk.forward(E[b], mode="chunk", chunk=chunk)
        logits = y @ W_out.T                       # logits[t] memprediksi byte s+t+1
        nxt = data[s + 1:e].astype(np.int64)       # byte YANG DIPREDIKSI
        if len(nxt) == 0:
            continue
        pr = np.clip(bit_probs(logits[:len(nxt)], nxt), 1e-6, 1 - 1e-6)
        out[s + 1:e] = np.log(pr / (1.0 - pr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=60000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--conv_k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_nn", action="store_true",
                    help="KONTROL: acak urutan masukan neural (kalau untung tetap ada -> bocor)")
    a = ap.parse_args()

    raw = np.frombuffer(load(CORPORA[a.data]), dtype=np.uint8)
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)} d={a.d} H={a.heads} "
          f"konv={a.conv_k} epoch={a.epochs}")

    print("\n[1] latih Tingkat B (byte CE, teacher-forced)")
    t0 = time.time()
    blk, E, W_out = train_tierB(tr, a.d, a.heads, a.conv_k, a.epochs, a.lr, a.batch, a.seed)
    print(f"    selesai dalam {time.time()-t0:.0f}s")

    # BPB mandiri Tingkat B pada jendela tahan
    yte, _ = blk.forward(E[te.astype(np.int64)], mode="chunk", chunk=128)
    lg = yte @ W_out.T
    tgt = np.concatenate([te[1:], te[:1]]).astype(np.int64)
    lo = lg - lg.max(1, keepdims=True)
    Ex = np.exp(lo)
    P = Ex / Ex.sum(1, keepdims=True)
    bpbB = float(-np.mean(np.log2(P[np.arange(len(te)), tgt] + 1e-300)))
    print(f"    Tingkat B mandiri (BPB tahan, ~1 langkah): {bpbB:.4f} bit")

    print("\n[2] hitung masukan neural untuk seluruh aliran")
    nn_tr = nn_stretch(blk, E, W_out, tr)
    nn_te = nn_stretch(blk, E, W_out, te)
    if a.shuffle_nn:                             # kontrol: hancurkan keselarasan temporal
        nn_tr = nn_tr[np.random.default_rng(0).permutation(len(nn_tr))]
        nn_te = nn_te[np.random.default_rng(0).permutation(len(nn_te))]
    print(f"    bentuk {nn_tr.shape}, |st| rata-rata {np.abs(nn_tr).mean():.3f}")

    print("\n[3] Tingkat A: tanpa vs dengan masukan neural")
    res = {}
    for tag, use_nn in (("A saja", False), ("A + B", True)):
        m = StatCore.from_config(Config.v2(stat_hash_bits=a.table_bits))
        for i, b in enumerate(tr):
            m.step(int(b), train=True, nn_st=(nn_tr[i] if use_nn else None))
        ce = m.eval_stream(te, nn_stream=(nn_te if use_nn else None))
        # ⚠️ B-01 (diperbaiki 2026-10-06): eval_stream mengembalikan BIT, bukan nat.
        # Versi lama membagi dengan LN2 -> terinflasi x1,4427. Jangan diulang.
        res[tag] = float(ce.mean())
        print(f"    {tag:<8}: {res[tag]:.4f} bit", flush=True)

    d = res["A saja"] - res["A + B"]
    print("\n== PUTUSAN G2 ==")
    print(f"  Tingkat A saja : {res['A saja']:.4f} bit")
    print(f"  Tingkat A + B  : {res['A + B']:.4f} bit")
    print(f"  selisih        : {d:+.4f} bit  (G2 menuntut >= +0,05)")
    print(f"  -> {'LULUS' if d >= 0.05 else 'BELUM LULUS'}")


if __name__ == "__main__":
    main()
